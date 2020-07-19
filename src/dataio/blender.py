import os
import json

import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF


def translate_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_along_x(phi=np.pi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_along_y(theta=np.pi):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_along_z(radius)
    c2w = rotate_along_x(phi * np.pi / 180.0) @ c2w
    c2w = rotate_along_y(theta * np.pi / 180.0)  @ c2w
    c2w = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]) @ c2w
    return c2w


def load(cfg):
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in ['train', 'val', 'test']:
        meta = None
        fname = os.path.join(cfg.dataset.basedir, 'transforms_{}.json'.format(s))
        with open(fname, 'r') as fp:
            meta = json.load(fp)

        imgs = []
        poses = []
        skip = 1 if s == 'train' or cfg.dataset.testskip < 1 else cfg.dataset.testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(cfg.dataset.basedir, frame['file_path'] + '.png')

            im = Image.open(fname)
            if cfg.dataset.half_res:
                im = TF.resize(im, (400, 400))
            imgs.append(np.array(im))

            poses.append(np.array(frame['transform_matrix']))

        # keep all 4 channels (RGBA)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs.shape[1:3]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40, endpoint=False)], axis=0)

    if cfg.nerf.train.white_background:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
    else:
        imgs = imgs[..., :3]

    hwf = [int(H), int(W), focal]

    print('Loaded blender', imgs.shape, poses.shape, render_poses.shape, hwf, cfg.dataset.basedir)
    return imgs, poses, render_poses, i_split, hwf
