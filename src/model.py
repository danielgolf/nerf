import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard

from utils import get_minibatches


def positional_encoding(x, num_freq, include_input=True):
    """
    positional encoding // kernel
    TODO: arguments mising here, but included in original implementation:
        * include_input (True)
        * log_sampling (True)
        * ...
    """
    freq = (x[..., None] * (2 ** torch.arange(num_freq).to(x))).view(x.shape[0], -1)
    if include_input:
        return torch.cat([x, torch.cos(freq), torch.sin(freq)], dim=-1)
    return torch.cat([torch.cos(freq), torch.sin(freq)], dim=-1)


class NerfMLP(nn.Module):
    def __init__(
            self,
            num_layers=8,
            num_neurons=256,
            dim_xyz=3,
            dim_dir=None,
            skips=None
        ):
        assert num_layers > 0
        assert num_neurons > 0
        assert dim_xyz > 0
        super().__init__()

        self.dim_xyz = dim_xyz
        self.dim_dir = dim_dir
        self.skips = skips if isinstance(skips, list) else []
        self.fc_base_list = nn.ModuleList()

        in_size = self.dim_xyz
        for i in range(num_layers):
            if i in self.skips:
                in_size += self.dim_xyz
            self.fc_base_list.append(
                nn.Linear(in_size, num_neurons)
            )
            in_size = num_neurons

        if dim_dir is not None:
            self.fc_alpha = nn.Linear(num_neurons, 1)
            self.fc_feat = nn.Linear(num_neurons, num_neurons)
            self.fc_last = nn.Linear(num_neurons + self.dim_dir, num_neurons // 2)
            self.fc_rgb = nn.Linear(num_neurons // 2, 3)
        else:
            self.fc_out = nn.Linear(num_neurons, 4)

    def forward(self, x):
        x_xyz = x[..., :self.dim_xyz]
        x_dir = None if self.dim_dir is None else x[..., self.dim_xyz:]

        x = x_xyz
        for i in range(len(self.fc_base_list)):
            if i in self.skips:
                x = torch.cat((x_xyz, x), -1)
            x = F.relu(self.fc_base_list[i](x))

        if x_dir is None:
            return self.fc_out(x)

        alpha = self.fc_alpha(x)

        x = self.fc_feat(x)
        x = torch.cat((x, x_dir), -1)
        x = F.relu(self.fc_last(x))
        rgb = self.fc_rgb(x)

        return torch.cat((rgb, alpha), -1)


class Nerf():
    def __init__(self, cfg):
        self.embed_xyz_coarse, self.embed_dir_coarse = None, None
        self.dim_xyz_coarse, self.dim_dir_coarse = None, None
        self.embed_xyz_fine, self.embed_dir_fine = None, None
        self.dim_xyz_fine, self.dim_dir_fine = None, None
        self._create_embeddings(cfg)

        self.model_coarse = NerfMLP(
            num_layers=cfg.model.coarse.num_layers,
            num_neurons=cfg.model.coarse.num_neurons,
            dim_xyz=self.dim_xyz_coarse,
            dim_dir=self.dim_dir_coarse,
            skips=cfg.model.coarse.skip_connections
        )

        self.model_fine = NerfMLP(
            num_layers=cfg.model.fine.num_layers,
            num_neurons=cfg.model.fine.num_neurons,
            dim_xyz=self.dim_xyz_fine,
            dim_dir=self.dim_dir_fine,
            skips=cfg.model.fine.skip_connections
        ) if cfg.model.fine is not None else None

        self.log_path = os.path.join(
            cfg.experiment.logdir,
            cfg.experiment.id
        )
        self.checkpoint_path = os.path.join(
            self.log_path,
            'checkpoint.pt'
        )

        self.opt, self.sched = None, None
        self.writer = tensorboard.writer.SummaryWriter(self.log_path)
        self.iter = 1
        print('Model loaded.')

    def _create_embeddings(self, cfg):
        self.embed_xyz_coarse = lambda x: positional_encoding(
            x, cfg.model.coarse.num_encoding_xyz,
            cfg.model.coarse.include_input_xyz
        )
        self.dim_xyz_coarse = 6 * cfg.model.coarse.num_encoding_xyz
        self.dim_xyz_coarse += 3 if cfg.model.coarse.include_input_xyz else 0

        if cfg.model.use_viewdirs:
            self.embed_dir_coarse = lambda x: positional_encoding(
                x, cfg.model.coarse.num_encoding_dir,
                cfg.model.coarse.include_input_dir
            )
            self.dim_dir_coarse = 6 * cfg.model.coarse.num_encoding_dir
            self.dim_dir_coarse += 3 if cfg.model.coarse.include_input_dir else 0

        if cfg.model.fine is None:
            return

        self.embed_xyz_fine = lambda x: positional_encoding(
            x, cfg.model.fine.num_encoding_xyz,
            cfg.model.fine.include_input_xyz
        )
        self.dim_xyz_fine = 6 * cfg.model.fine.num_encoding_xyz
        self.dim_xyz_fine += 3 if cfg.model.fine.include_input_xyz else 0

        if cfg.model.use_viewdirs:
            self.embed_dir_fine = lambda x: positional_encoding(
                x, cfg.model.fine.num_encoding_dir,
                cfg.model.fine.include_input_dir
            )
            self.dim_dir_fine = 6 * cfg.model.fine.num_encoding_dir
            self.dim_dir_fine += 3 if cfg.model.fine.include_input_dir else 0

    def create_optimizer(self, cfg):
        params = list(self.model_coarse.parameters())
        if self.model_fine is not None:
            params += list(self.model_fine.parameters())

        self.opt = getattr(torch.optim, cfg.train.optimizer.type)(
            params,
            cfg.train.optimizer.lr
        )
        dec_fac = cfg.train.scheduler.lr_decay_factor
        decay = cfg.train.scheduler.lr_decay * 1000
        self.sched = torch.optim.lr_scheduler.ExponentialLR(
            self.opt, dec_fac ** (1 / decay)
        )

    def load(self):
        if not os.path.isfile(self.checkpoint_path):
            print('No checkpoint found.')
            return

        checkpoint = torch.load(self.checkpoint_path)
        self.model_coarse.load_state_dict(checkpoint['model_coarse'])
        if 'model_fine' in checkpoint:
            self.model_fine.load_state_dict(checkpoint['model_fine'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.sched.load_state_dict(checkpoint['sched'])
        self.iter = checkpoint['iter']

    def save(self):
        checkpoint = {
            'model_coarse': self.model_coarse.state_dict(),
            'opt': self.opt.state_dict(),
            'sched': self.sched.state_dict(),
            'iter': self.iter
        }
        if self.model_fine is None:
            checkpoint['model_fine'] = None
        else:
            checkpoint['model_fine'] = self.model_fine.state_dict()

        torch.save(checkpoint, self.checkpoint_path)

    def to(self, device):
        self.model_coarse.to(device)
        if self.model_fine is not None:
            self.model_fine.to(device)

    def train(self):
        self.model_coarse.train()
        if self.model_fine is not None:
            self.model_fine.train()

    def eval(self):
        self.model_coarse.eval()
        if self.model_fine is not None:
            self.model_fine.eval()

    def predict_coarse(self, x_xyz, ray_dir=None, chunksize=8192):
        x = self.embed_xyz_coarse(x_xyz.reshape((-1, x_xyz.shape[-1])))
        if self.embed_dir_coarse is not None:
            x_dir = ray_dir / ray_dir.norm(p=2, dim=-1).unsqueeze(-1)
            x_dir = x_dir[..., None, :].expand(x_xyz.shape)
            x_dir = self.embed_dir_coarse(x_dir.reshape((-1, x_dir.shape[-1])))
            x = torch.cat((x, x_dir), dim=-1)

        batches = get_minibatches(x, chunksize)
        out = torch.cat([self.model_coarse(xin) for xin in batches], dim=0)
        return out.reshape(x_xyz.shape[:-1] + out.shape[-1:])

    def predict_fine(self, x_xyz, ray_dir=None, chunksize=8192):
        x = self.embed_xyz_fine(x_xyz.reshape((-1, x_xyz.shape[-1])))
        if self.embed_dir_fine is not None:
            x_dir = ray_dir / ray_dir.norm(p=2, dim=-1).unsqueeze(-1)
            x_dir = x_dir[..., None, :].expand(x_xyz.shape)
            x_dir = self.embed_dir_fine(x_dir.reshape((-1, x_dir.shape[-1])))
            x = torch.cat((x, x_dir), dim=-1)

        batches = get_minibatches(x, chunksize)
        out = torch.cat([self.model_fine(xin) for xin in batches], dim=0)
        return out.reshape(x_xyz.shape[:-1] + out.shape[-1:])
