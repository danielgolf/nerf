import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image

import torchsearchsorted


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


# TODO understand and rewrite
def sample_pdf(bins, weights, num_samples, det=False):
    """
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [num_samples],
            dtype=weights.dtype,
            device=weights.device,
        )

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()

    # TODO what does that?
    inds = torchsearchsorted.searchsorted(cdf, u, side="right")

    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(to_pil_image(tensor.permute(2, 0, 1).cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img
