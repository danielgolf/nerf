import torch
import numpy as np
from tqdm import trange

from model import Nerf
from dataio.loader import load_data


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


def get_minibatches(inputs, chunksize=8192):
    out = []
    for i in range(0, inputs.shape[0], chunksize):
        out.append(inputs[i:i+chunksize])
    return out


def train(cfg):
    # Device on which to run
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # TODO: make consistent with other data types (e.g. llff)
    # Load data
    near, far = cfg.dataset.near, cfg.dataset.far
    images, poses, render_poses, i_split, hwf = load_data(cfg)
    i_train, i_val, i_test = i_split
    H, W, focal = hwf

    # Create nerf model
    nerf = Nerf(cfg)
    nerf.load()
    nerf.to(device)

    for _ in trange(nerf.iter, cfg.train.iters):
        nerf.train()    # require gradients and stuff

        idx = np.random.choice(i_train)
        img = torch.from_numpy(images[idx]).to(device)
        pose = torch.from_numpy(poses[idx]).to(device)

        cords, ray_ori, ray_dir= get_ray_bundle(
            H, W, focal, pose
        )

        ray_idx = np.random.choice( # take a subset of all rays
            cords.shape[0],
            size=cfg.train.num_random_rays,
            replace=False
        )

        cords = cords[ray_idx]
        ray_ori = ray_ori[cords[:, 0], cords[:, 1]]
        ray_dir = ray_dir[cords[:, 0], cords[:, 1]]
        img = img[cords[:, 0], cords[:, 1]]

        # TODO NDC option

        near = near * torch.ones_like(ray_ori[..., :1])
        far = far * torch.ones_like(ray_ori[..., :1])

        t = torch.linspace(0., 1., cfg.train.num_coarse).to(pose)
        # TODO: lindisp option
        z_vals = near * (1. - t) + far * t
        # TODO: maybe needed for LLFF/deepvoxel dataset
        # z_vals = z_vals.expand([ray_batch.shape[0], cfg.train.num_coarse])
        # basically eq. 2 in the paper
        if cfg.train.perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            rand = torch.rand(z_vals.shape).to(z_vals)
            z_vals = lower + (upper - lower) * rand

        x_xyz = ray_ori[..., None, :] + ray_dir[..., None, :] * z_vals[..., :, None]
        x = nerf.embed_xyz_coarse(x_xyz.reshape((-1, x_xyz.shape[-1])))
        if cfg.model.use_viewdirs:
            x_dir = ray_dir / ray_dir.norm(p=2, dim=-1).unsqueeze(-1)
            # TODO: maybe needed for LLFF/deepvoxel dataset
            # viewdirs = viewdirs.view((-1, 3))
            x_dir = x_dir[..., None, :].expand(x_xyz.shape)
            x_dir = nerf.embed_dir_coarse(x_dir.reshape((-1, x_dir.shape[-1])))
            x = torch.cat((x, x_dir), dim=-1)

        batches = get_minibatches(x, cfg.train.chunksize)
        out = torch.cat([nerf.model_coarse(xin) for xin in batches], dim=0)
        out = out.reshape(x_xyz.shape[:-1] + out.shape[-1:])

        # === dev ===
