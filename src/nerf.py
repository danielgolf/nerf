import torch

from render import volume_render_radiance_field
from utils import sample_pdf, get_minibatches


def get_ray_bundle(height, width, focal_length, tform_cam2world):
    ii, jj = torch.meshgrid(
        torch.arange(width).to(tform_cam2world),
        torch.arange(height).to(tform_cam2world)
    )
    xx, yy = ii.T, jj.T
    directions = torch.stack([
        (xx - width * 0.5) / focal_length,
        -(yy - height * 0.5) / focal_length,
        -torch.ones_like(xx),
    ], -1)
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    coords = torch.stack([yy, xx], -1).to(torch.int64).reshape(-1, 2)
    return coords, ray_origins, ray_directions


def nerf_iteration(nerf, cfg, rays, near_val, far_val, mode='train'):
    batches = get_minibatches(rays, getattr(cfg, mode).chunksize)
    coarse, fine = [], []
    for batch in batches:
        ray_ori, ray_dir = batch[..., :3], batch[..., 3:]

        # TODO NDC option
        near = near_val * torch.ones_like(ray_ori[..., :1])
        far = far_val * torch.ones_like(ray_ori[..., :1])
        t = torch.linspace(0., 1., getattr(cfg, mode).num_coarse).to(ray_ori)
        # TODO: lindisp option
        z_vals = near * (1. - t) + far * t

        # basically eq. 2 in the paper
        if getattr(cfg, mode).perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            rand = torch.rand(z_vals.shape).to(z_vals)
            z_vals = lower + (upper - lower) * rand

        x_xyz = ray_ori[..., None, :] + ray_dir[..., None, :] * z_vals[..., :, None]
        out = nerf.predict_coarse(x_xyz, ray_dir, getattr(cfg, mode).chunksize)

        rgb_coarse, weights = volume_render_radiance_field(
            out, z_vals, ray_dir,
            getattr(cfg, mode).radiance_noise_std,
            getattr(cfg, mode).white_background
        )
        coarse.append(rgb_coarse)

        if nerf.model_fine is not None:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid,
                weights[..., 1:-1],
                getattr(cfg, mode).num_fine,
                det=(getattr(cfg, mode).perturb == 0.0),
            )

            # important: backprop fine loss online to fine network
            # otherwise the sampling is backpropagated to the coarse network
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)

            x_xyz = ray_ori[..., None, :] + ray_dir[..., None, :] * z_vals[..., :, None]
            out = nerf.predict_fine(x_xyz, ray_dir, getattr(cfg, mode).chunksize)

            rgb_fine, _ = volume_render_radiance_field(
                out, z_vals, ray_dir,
                getattr(cfg, mode).radiance_noise_std,
                getattr(cfg, mode).white_background
            )
            fine.append(rgb_fine)

    if nerf.model_fine is None:
        return torch.cat(coarse, dim=0), None
    return torch.cat(coarse, dim=0), torch.cat(fine, dim=0)
