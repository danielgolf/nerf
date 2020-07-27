import torch

import torchsearchsorted


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


def gather_cdf_util(cdf, inds):
    orig_inds_shape = inds.shape
    inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
    valid_mask = [
        torch.where(ind >= cdf.shape[1], torch.zeros_like(
            ind), torch.ones_like(ind))
        for ind in inds_flat
    ]
    inds_flat = [
        torch.where(
            ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
        for ind in inds_flat
    ]
    cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
    cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
    cdf_flat = [
        cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
    ]
    return torch.cat(cdf_flat, dim=0)


def sample_pdf(bins, weights, num_samples, deterministic=False):
    weights = weights + 1e-5  # prevent nans
    pdf = weights / weights.sum(-1).unsqueeze(-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

    # Take uniform samples
    if deterministic:
        u = torch.linspace(0.0, 1.0, num_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)

    # Invert CDF
    inds = torchsearchsorted.searchsorted(
        cdf.contiguous(), u.contiguous(), side="right"
    )

    below = torch.max(torch.zeros_like(inds), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)

    cdf_g = gather_cdf_util(cdf, inds_g)
    bins_g = gather_cdf_util(bins, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    return tensor.detach().cpu().permute(2, 0, 1).numpy()
