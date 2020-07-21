import torch

from model import Nerf
from dataio.loader import load_data


def train(cfg):
    # TODO: make consistent with other data types (e.g. llff)
    # Load data
    near, far = cfg.dataset.near, cfg.dataset.far
    images, poses, render_poses, i_split, hwf = load_data(cfg)
    H, W, focal = hwf

    # Device on which to run
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Create nerf model
    nerf = Nerf(cfg)
    nerf.load()
    nerf.to(device)
