import torch


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


def cumprod_exclusive(tensor):
    """
    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    """
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.0
    return cumprod

