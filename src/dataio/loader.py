import torch
import numpy as np

from dataio import blender
from nerf import get_ray_bundle


class NerfData():
    def __init__(self, cfg):
        self.near = cfg.dataset.near
        self.far = cfg.dataset.far

        data = self._call_loader(cfg)

        # TODO: make consistent with other data types (e.g. llff)
        # TODO: what do I actually need here?
        self.images, self.poses, self.render_poses = data[:3]
        self.i_train, self.i_val, self.i_test = data[3]
        self.height, self.width, self.focal = data[4]

    def _call_loader(self, cfg):
        if cfg.dataset.type == 'blender':
            return blender.load(cfg)

        if cfg.dataset.type == 'llff':
            print("LLFF data not supported yet")
            quit(1)

        if cfg.dataset.type == 'deepvoxels':
            print("Deepvoxel data not supported yet")
            quit(1)

        print('Unknown dataset type', cfg.dataset.type, 'exiting')
        quit(1)

    def to(self, device):
        self.images.to(device)
        self.poses.to(device)

    def get_train_batch(self, size=1024):
        idx = np.random.choice(self.i_train)
        img = self.images[idx]
        pose = self.poses[idx]

        coords, ray_ori, ray_dir = get_ray_bundle(
            self.height, self.width, self.focal, pose
        )

        ray_idx = np.random.choice( # take a subset of all rays
            coords.shape[0],
            size=size,
            replace=False
        )

        coords = coords[ray_idx]
        ray_ori = ray_ori[coords[:, 0], coords[:, 1]]
        ray_dir = ray_dir[coords[:, 0], coords[:, 1]]
        img = img[coords[:, 0], coords[:, 1]]

        rays = torch.cat([ray_ori, ray_dir], dim=-1)
        return rays, img

    def get_valid_img(self):
        idx = np.random.choice(self.i_val)
        img = self.images[idx]
        pose = self.poses[idx]

        _, ray_ori, ray_dir= get_ray_bundle(
            self.height, self.width, self.focal, pose
        )

        ray_ori = ray_ori.reshape((-1, ray_ori.shape[-1]))
        ray_dir = ray_dir.reshape((-1, ray_dir.shape[-1]))

        rays = torch.cat([ray_ori, ray_dir], dim=-1)
        return rays, img
