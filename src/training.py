import torch
import numpy as np
from tqdm import trange, tqdm

from model import Nerf
from dataio.loader import load_data
from render import volume_render_radiance_field
from utils import get_ray_bundle, sample_pdf, cast_to_image


def nerf_iteration(nerf, cfg, pose, ray_ori, ray_dir, near, far, mode='train'):
    # TODO NDC option
    near = near * torch.ones_like(ray_ori[..., :1])
    far = far * torch.ones_like(ray_ori[..., :1])
    t = torch.linspace(0., 1., getattr(cfg, mode).num_coarse).to(pose)
    # TODO: lindisp option
    z_vals = near * (1. - t) + far * t
    # TODO: maybe needed for LLFF/deepvoxel dataset
    # z_vals = z_vals.expand([ray_batch.shape[0], cfg.train.num_coarse])
    del near, far, t
    torch.cuda.empty_cache()

    # basically eq. 2 in the paper
    if getattr(cfg, mode).perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * rand
        del mids, upper, lower, rand
        torch.cuda.empty_cache()

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
        del z_vals_mid, z_samples, out
        torch.cuda.empty_cache()

        x_xyz = ray_ori[..., None, :] + ray_dir[..., None, :] * z_vals[..., :, None]
        out = nerf.run(x_xyz, ray_dir, getattr(cfg, mode).chunksize)

        rgb_fine, _ = volume_render_radiance_field(
            out, z_vals, ray_dir,
            getattr(cfg, mode).radiance_noise_std, getattr(cfg, mode).white_background
        )

    return rgb_coarse, rgb_fine


def train(cfg):
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

    # Create nerf model
    nerf = Nerf(cfg)
    nerf.load()
    nerf.to(device)

    for i in trange(nerf.iter, cfg.train.iters):
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

        rgb_coarse, rgb_fine = nerf_iteration(
            nerf,
            cfg,
            pose,
            ray_ori,
            ray_dir,
            near,
            far
        )

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], img[..., :3]
        )

        fine_loss = 0.0
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], img[..., :3]
            )

        loss = coarse_loss + fine_loss
        loss.backward()
        nerf.opt.step()
        nerf.sched.step()
        nerf.opt.zero_grad()

        printA = i % cfg.train.print_every == 0
        printB = i == cfg.train.iters - 1
        if printA or printB:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")

        saveA = i % cfg.train.save_every == 0
        if saveA or printB:
            nerf.save()
            tqdm.write("=== Saved Checkpoint ===")

        nerf.writer.add_scalar("train/loss", loss.item(), i)
        nerf.writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        nerf.writer.add_scalar("train/lr", nerf.sched.get_last_lr()[0], i)
        if rgb_fine is not None:
            nerf.writer.add_scalar("train/fine_loss", fine_loss.item(), i)

        validA = i % cfg.validation.every
        if validA or printB:
            tqdm.write(f"[VAL] ===> Iter: {i}")
            nerf.eval()

            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None

                idx = np.random.choice(i_train)
                img = torch.from_numpy(images[idx]).to(device)
                pose = torch.from_numpy(poses[idx]).to(device)

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
                    mode="validation"
                )

                rgb_coarse = rgb_coarse.reshape(img[..., :3].shape)
                if rgb_fine is not None:
                    rgb_fine = rgb_fine.reshape(img[..., :3].shape)

                coarse_loss = torch.nn.functional.mse_loss(
                    rgb_coarse[..., :3], img[..., :3]
                )

                fine_loss = 0.0
                if rgb_fine is not None:
                    fine_loss = torch.nn.functional.mse_loss(
                        rgb_fine[..., :3], img[..., :3]
                    )

                loss = coarse_loss + fine_loss

                nerf.writer.add_scalar("validation/loss", loss.item(), i)
                nerf.writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                nerf.writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                nerf.writer.add_image(
                    "validation/img_target",
                    cast_to_image(img[..., :3]),
                    i
                )
                if rgb_fine is not None:
                    nerf.writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    nerf.writer.add_scalar("validation/fine_loss",
                                      fine_loss.item(), i)

                tqdm.write(f"[VALID] Iter: {i} Loss: {loss.item()}")

    print("Done!")