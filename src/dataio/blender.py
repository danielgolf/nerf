import os
import json

import torch
from math import pi
from PIL import Image
import torchvision.transforms.functional as TF


def translate_along_z(t):
    tform = torch.eye(4)
    tform[2, 3] = t
    return tform


def rotate_along_x(phi=1.):
    phi = torch.Tensor([phi])
    tform = torch.eye(4)
    tform[1, 1] = tform[2, 2] = torch.cos(phi)
    tform[2, 1] = torch.sin(phi)
    tform[1, 2] = -tform[2, 1]
    return tform


def rotate_along_y(theta=1.):
    theta = torch.Tensor([theta])
    tform = torch.eye(4)
    tform[0, 0] = tform[2, 2] = torch.cos(theta)
    tform[2, 0] = torch.sin(theta)
    tform[0, 2] = -tform[2, 0]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_along_z(radius)
    c2w = rotate_along_x(phi * pi / 180.).matmul(c2w)
    c2w = rotate_along_y(theta * pi / 180.).matmul(c2w)
    c2w = torch.Tensor([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]).matmul(c2w)
    return c2w


def load(cfg):
    all_imgs, all_poses = [], []
    counts = [0]

    for s in ['train', 'val', 'test']:
        meta = None
        fname = os.path.join(cfg.dataset.path, 'transforms_{}.json'.format(s))
        with open(fname, 'r') as fp:
            meta = json.load(fp)

        imgs = []
        poses = []
        if s == 'train' or cfg.dataset.testskip < 2:
            skip = 1
        else:
            skip = cfg.dataset.testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(cfg.dataset.path, frame['file_path'] + '.png')

            im = Image.open(fname)
            if cfg.dataset.half_res:
                im = TF.resize(im, (400, 400))
            imgs.append(TF.to_tensor(im))
            poses.append(torch.Tensor(frame['transform_matrix']))

        # keep all 4 channels (RGBA)
        imgs = torch.stack(imgs, dim=0).permute(0, 2, 3, 1)
        poses = torch.stack(poses, dim=0)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [torch.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = torch.cat(all_imgs, dim=0)
    poses = torch.cat(all_poses, dim=0)

    H, W = imgs.shape[1:3]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / torch.tan(torch.Tensor([.5 * camera_angle_x]))[0]

    render_poses = torch.stack([
        pose_spherical(angle, -30., 4.)
        for angle in torch.linspace(-180., 180., 41)[:-1]
    ], dim=0)

    if cfg.train.white_background:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
    else:
        imgs = imgs[..., :3]

    hwf = [int(H), int(W), focal]

    print(
        'Loaded blender',
        cfg.dataset.path,
        imgs.shape,
        poses.shape,
        render_poses.shape,
        hwf
    )
    return imgs, poses, render_poses, i_split, hwf
