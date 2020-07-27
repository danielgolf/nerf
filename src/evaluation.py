import os
import torch
import imageio
from tqdm import tqdm

from model import Nerf
from dataio.loader import load_data
from render import volume_render_radiance_field
from utils import get_ray_bundle, sample_pdf


def nerf_iteration(nerf, cfg, pose, ray_ori, ray_dir, near, far, mode='train'):
    # TODO NDC option
    near = near * torch.ones_like(ray_ori[..., :1])
    far = far * torch.ones_like(ray_ori[..., :1])

    t = torch.linspace(0., 1., getattr(cfg, mode).num_coarse).to(pose)
    # TODO: lindisp option
    z_vals = near * (1. - t) + far * t
    # TODO: maybe needed for LLFF/deepvoxel dataset
    # z_vals = z_vals.expand([ray_batch.shape[0], cfg.train.num_coarse])
    # basically eq. 2 in the paper
    if getattr(cfg, mode).perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * rand

    x_xyz = ray_ori[..., None, :] + ray_dir[..., None, :] * z_vals[..., :, None]
    out = nerf.run(x_xyz, ray_dir, getattr(cfg, mode).chunksize)

    rgb_coarse, weights = volume_render_radiance_field(
        out, z_vals, ray_dir,
        getattr(cfg, mode).radiance_noise_std, getattr(cfg, mode).white_background
    )

    rgb_fine = None, None, None
    if nerf.model_fine is not None:
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(cfg, mode).num_fine,
            deterministic=(getattr(cfg, mode).perturb == 0.0),
        )
        # TODO: needed? z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)

        x_xyz = ray_ori[..., None, :] + ray_dir[..., None, :] * z_vals[..., :, None]
        out = nerf.run(x_xyz, ray_dir, getattr(cfg, mode).chunksize)

        rgb_fine, _ = volume_render_radiance_field(
            out, z_vals, ray_dir,
            getattr(cfg, mode).radiance_noise_std, getattr(cfg, mode).white_background
        )

    return rgb_coarse, rgb_fine


def eval(cfg):
    expid = cfg.experiment.id
    logdir = cfg.experiment.logdir
    save_dir = os.path.join(logdir, expid, 'rendered')
    os.makedirs(save_dir, exist_ok=True)

    # Device on which to run
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # TODO: make consistent with other data types (e.g. llff)
    # Load data
    # TODO: what do I actually need here?
    near, far = cfg.dataset.near, cfg.dataset.far
    images, poses, render_poses, i_split, hwf = load_data(cfg)
    i_train, i_val, i_test = i_split
    H, W, focal = hwf

    img_shape = images[0][..., :3].shape

    # Create nerf model
    nerf = Nerf(cfg)
    nerf.load()
    nerf.to(device)
    nerf.eval()

    with torch.no_grad():
        for i, pose in enumerate(tqdm(render_poses)):
            _, ray_ori, ray_dir= get_ray_bundle(
                H, W, focal, pose
            )

            # TODO Make 3d rays possible in nerf_iteration
            ray_ori = ray_ori.reshape((-1, ray_ori.shape[-1]))
            ray_dir = ray_dir.reshape((-1, ray_dir.shape[-1]))

            rgb_coarse, rgb_fine = nerf_iteration(
                nerf,
                cfg,
                pose,
                ray_ori,
                ray_dir,
                near,
                far,
                mode='validation'
            )

            if rgb_fine is not None:
                rgb = rgb_fine.reshape(img_shape)
            else:
                rgb = rgb_coarse.reshape(img_shape)

            save_file = os.path.join(save_dir, f"{i:04d}.png")
            imageio.imwrite(
                save_file,
                rgb.detach().cpu().numpy()
            )
