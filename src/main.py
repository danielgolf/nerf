import os
import torch
import imageio
import numpy as np
from tqdm import trange, tqdm
from torchvision.transforms.functional import to_pil_image

from model import NerfMLP
from utils import cast_to_image
from configuration import getcfg
from dataio.loader import NerfData
from nerf import nerf_iteration, get_ray_bundle


def setup_dirs(cfg):
    expid = cfg.experiment.id
    logdir = cfg.experiment.logdir
    os.makedirs(os.path.join(logdir, expid), exist_ok=True)

    fname = os.path.join(logdir, expid, 'config.yml')
    with open(fname, 'w') as fp:
        fp.write(open(cfg.configuration_path, 'r').read())


def fix_seed(seed):
    if seed is not None:
        print('Fixing random seed:', seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def evaluation(cfg, mlp, data):
    expid = cfg.experiment.id
    logdir = cfg.experiment.logdir
    save_dir = os.path.join(logdir, expid, 'rendered')
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, pose in enumerate(tqdm(data.render_poses)):
            _, ray_ori, ray_dir = get_ray_bundle(
                data.height, data.width, data.focal, pose
            )
            img_shape = ray_ori.shape
            ray_ori = ray_ori.reshape((-1, ray_ori.shape[-1]))
            ray_dir = ray_dir.reshape((-1, ray_dir.shape[-1]))

            rgb_coarse, rgb_fine = nerf_iteration(
                mlp,
                cfg,
                torch.cat([ray_ori, ray_dir], dim=-1),
                data.near,
                data.far,
                mode='validation'
            )

            if rgb_fine is not None:
                rgb = rgb_fine.reshape(img_shape)
            else:
                rgb = rgb_coarse.reshape(img_shape)

            save_file = os.path.join(save_dir, f"{i:04d}.png")
            img = np.array(to_pil_image(rgb.permute(2, 0, 1).cpu()))
            imageio.imwrite(save_file, img)


def train(cfg, mlp, data):
    train_losses = []
    for i in trange(mlp.iter, cfg.train.iters + 1):
        mlp.train()    # require gradients and stuff

        rays, img = data.get_train_batch(cfg.train.num_random_rays)
        rgb_coarse, rgb_fine = nerf_iteration(
            mlp,
            cfg,
            rays,
            data.near,
            data.far
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
        mlp.opt.step()
        mlp.sched.step()
        mlp.opt.zero_grad()

        lastIt = i == cfg.train.iters - 1
        printIt = i % cfg.train.print_every == 0
        if lastIt or printIt:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")

        saveIt = i % cfg.train.save_every == 0
        if lastIt or saveIt:
            mlp.iter = i
            mlp.save()
            tqdm.write("=== Saved Checkpoint ===")

        train_losses.append(loss.item())
        mlp.writer.add_scalar("train/loss", loss.item(), i)
        mlp.writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        mlp.writer.add_scalar("train/lr", mlp.sched.get_last_lr()[0], i)
        if rgb_fine is not None:
            mlp.writer.add_scalar("train/fine_loss", fine_loss.item(), i)

        valIt = i % cfg.validation.every == 0
        if lastIt or valIt:
            mlp.eval()

            with torch.no_grad():
                rays, img = data.get_valid_img()
                rgb_coarse, rgb_fine = nerf_iteration(
                    mlp,
                    cfg,
                    rays,
                    data.near,
                    data.far,
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

                mlp.writer.add_scalars("both/loss", {
                    "train": sum(train_losses) / len(train_losses),
                    "valid": loss.item()
                })
                train_losses = []
                mlp.writer.add_scalar("validation/loss", loss.item(), i)
                mlp.writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                mlp.writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                mlp.writer.add_image(
                    "validation/img_target",
                    cast_to_image(img[..., :3]),
                    i
                )
                if rgb_fine is not None:
                    mlp.writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    mlp.writer.add_scalar(
                        "validation/fine_loss", fine_loss.item(), i
                    )

                tqdm.write(f"[VALID] Iter: {i} Loss: {loss.item()}")
    print("Done!")


def main(cfg):
    setup_dirs(cfg)
    fix_seed(cfg.experiment.random_seed)

    # Device on which to run
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Running on device: {device.replace('cuda', 'gpu').upper()}")

    # torch.autograd.set_detect_anomaly(True)     # debugging

    # Create nerf model
    mlp = NerfMLP(cfg)
    mlp.to(device)
    mlp.create_optimizer(cfg)
    mlp.load()

    # Load data
    data = NerfData(cfg)
    data.to(device)

    if cfg.render_only:
        evaluation(cfg, mlp, data)
        return

    train(cfg, mlp, data)


if __name__ == "__main__":
    main(getcfg())
