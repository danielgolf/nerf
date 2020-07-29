import os
import torch
import imageio
import torchvision
import numpy as np
from tqdm import tqdm

from model import Nerf
from utils import get_ray_bundle
from training import nerf_iteration
from dataio.loader import load_data


def evalnerf(cfg):
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
    _, _, render_poses, _, hwf = load_data(cfg)
    H, W, focal = hwf

    # Create nerf model
    nerf = Nerf(cfg)
    nerf.load()
    nerf.to(device)
    nerf.eval()

    with torch.no_grad():
        for i, pose in enumerate(tqdm(render_poses)):
            pose = torch.from_numpy(pose).to(torch.float32).to(device)
            _, ray_ori, ray_dir = get_ray_bundle(
                H, W, focal, pose
            )

            img_shape = ray_ori.shape
            ray_ori = ray_ori.reshape((-1, ray_ori.shape[-1]))
            ray_dir = ray_dir.reshape((-1, ray_dir.shape[-1]))

            rgb_coarse, rgb_fine = nerf_iteration(
                nerf,
                cfg,
                torch.cat([ray_ori, ray_dir], dim=-1),
                near,
                far,
                mode='validation'
            )

            if rgb_fine is not None:
                rgb = rgb_fine.reshape(img_shape)
            else:
                rgb = rgb_coarse.reshape(img_shape)

            save_file = os.path.join(save_dir, f"{i:04d}.png")
            img = np.array(torchvision.transforms.ToPILImage()(
                rgb.permute(2, 0, 1).detach().cpu()))
            imageio.imwrite(save_file, img)
