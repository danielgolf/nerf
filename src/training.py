import torch
import numpy as np
from tqdm import trange

from model import Nerf
from utils import get_ray_bundle
from dataio.loader import load_data
from render import volume_render_radiance_field


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
        out = nerf.run(x_xyz, ray_dir, cfg.train.chunksize)

        rgb_map, disp_map, acc_map, weights, depth_map = volume_render_radiance_field(
            out, z_vals, ray_dir,
            cfg.train.radiance_noise_std, cfg.train.white_background
        )

        rgb_fine, disp_fine, acc_fine = None, None, None

        if nerf.model_fine is not None:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid,
                weights[..., 1:-1],
                cfg.train.num_fine,
                deterministic=(cfg.train.perturb == 0.0),
            )
            # TODO: needed? z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)

            x_xyz = ray_ori[..., None, :] + ray_dir[..., None, :] * z_vals[..., :, None]
            out = nerf.run(x_xyz, ray_dir, cfg.train.chunksize)

            rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(
                out, z_vals, ray_dir,
                cfg.train.radiance_noise_std, cfg.train.white_background
            )