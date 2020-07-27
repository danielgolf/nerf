import os
import torch
import numpy as np

from training import train
from evaluation import evalnerf
from configuration import getcfg


def setup_dirs(cfg):
    """
    Create log dir and copy the config file
    """

    expid = cfg.experiment.id
    logdir = cfg.experiment.logdir
    os.makedirs(os.path.join(logdir, expid), exist_ok=True)

    fname = os.path.join(logdir, expid, 'config.yml')
    with open(fname, 'w') as fp:
        fp.write(open(cfg.configuration_path, 'r').read())


def fix_seed(seed):
    if seed is not None:
        print('Fixing random seed', seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def main(cfg):
    setup_dirs(cfg)
    fix_seed(cfg.experiment.random_seed)

    if cfg.render_only:
        evalnerf(cfg)
        return

    train(cfg)


if __name__ == "__main__":
    main(getcfg())
