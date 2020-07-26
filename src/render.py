import torch

from utils import cumprod_exclusive


def volume_render_radiance_field(
        out,
        z_vals,
        ray_dir,
        std_noise=0.0,
        white_background=False,
    ):

    rgb = torch.sigmoid(out[..., :3])

    dists = torch.cat((
        z_vals[..., 1:] - z_vals[..., :-1],
        1e10 * torch.ones_like(z_vals[..., :1])
    ), dim=-1)
    dists = dists * ray_dir[..., None, :].norm(p=2, dim=-1)

    noise = 0.0
    if std_noise > 0.0:
        noise = torch.randn(out[..., 3].shape) * std_noise
        noise = noise.to(out)

    sigma_a = torch.nn.functional.relu(out[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)

    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)

    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    disp_map = 1 / disp_map

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, weights
