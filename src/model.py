import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard


class Embedder:
    """
    positional encoding // kernel
    TODO: arguments mising here, but included in original implementation:
        * include_input (True)
        * log_sampling (True)
        * ...
    """

    def __init__(self, num_freqs, periodic_fns, include_input=True):
        self.num_freqs = num_freqs
        self.periodic_fns = periodic_fns
        self.create_embedding_fn(include_input)

    def create_embedding_fn(self, include_input):
        n_fns = 0
        embed_fns = []

        if include_input:
            n_fns += 1
            embed_fns.append(lambda x: x)

        freq_bands = 2. ** torch.arange(self.num_freqs)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                n_fns += 1
                # TODO: original paper states 2^k * pi * x - where is the pi?
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))

        self.embed_fns = embed_fns
        self.num_fns = n_fns

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


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

        self.skips = [4]
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

        self._create_optimizer(cfg)

        self.log_path = os.path.join(
            cfg.experiment.logdir,
            cfg.experiment.id
        )
        self.checkpoint_path = os.path.join(
            self.log_path,
            'checkpoint.pt'
        )
        self.writer = tensorboard.writer.SummaryWriter(self.log_path)
        self.iter = 0
        print('Model loaded.')

    def _create_embeddings(self, cfg):
        periodic_fns = [torch.sin, torch.cos]

        e = Embedder(
            cfg.model.coarse.num_encoding_xyz,
            periodic_fns,
            cfg.model.coarse.include_input_xyz
        )
        self.embed_xyz_coarse = lambda x, emb=e: emb.embed(x)
        self.dim_xyz_coarse = 3 * e.num_fns

        if cfg.model.use_viewdirs:
            e = Embedder(
                cfg.model.coarse.num_encoding_dir,
                periodic_fns,
                cfg.model.coarse.include_input_dir
            )
            self.embed_dir_coarse = lambda x, emb=e: emb.embed(x)
            self.dim_dir_coarse = 3 * e.num_fns

        if cfg.model.fine is None:
            return

        e = Embedder(
            cfg.model.fine.num_encoding_xyz,
            periodic_fns,
            cfg.model.fine.include_input_xyz
        )
        self.embed_xyz_fine = lambda x, emb=e: emb.embed(x)
        self.dim_xyz_fine = 3 * e.num_fns

        if cfg.model.use_viewdirs:
            e = Embedder(
                cfg.model.fine.num_encoding_dir,
                periodic_fns,
                cfg.model.fine.include_input_dir
            )
            self.embed_dir_fine = lambda x, emb=e: emb.embed(x)
            self.dim_dir_fine = 3 * e.num_fns

    def _create_optimizer(self, cfg):
        # TODO exponential decay
        params = list(self.model_coarse.parameters())
        if self.model_fine is not None:
            params += list(self.model_fine.parameters())

        self.opt = getattr(torch.optim, cfg.train.optimizer.type)(
            self.model_coarse.parameters(),
            cfg.train.optimizer.lr
        )

    def load(self):
        if not os.path.isfile(self.checkpoint_path):
            print('No checkpoint found.')
            return

        checkpoint = torch.load(self.checkpoint_path)
        self.model_coarse.load_state_dict(checkpoint['model_coarse'])
        if checkpoint['model_fine'] is not None:
            self.model_fine.load_state_dict(checkpoint['model_fine'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.iter = checkpoint['iter']

    def save(self):
        checkpoint = {
            'model_coarse': self.model_coarse.state_dict(),
            'opt': self.opt.state_dict(),
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
