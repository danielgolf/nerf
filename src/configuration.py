import argparse
import yaml


class Config(dict):
    """
    Fancy config object (inspired by nerf-pytorch implementation).
    Makes it possible to access a dictionary by member attributes.
    Example:
        cfg = Config({'a': 1, 'b': {'c': 2}})
        print(cfg.a) --> 1
        print(cfg.b.c) --> 2
    """
    @classmethod
    def _create_config_tree(cls, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = cls(v)
        return d

    def __init__(self, d):
        d = self._create_config_tree(d)
        super().__init__(d)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            # TODO: change back, but for now make sure that all
            # attributes used are in the config file
            raise AttributeError("No such attribute: " + name)
            # return None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def getcfg():
    """
    Asserts that config file is given as a command line parameter.
    Config.yml is loaded and custom config object is returned.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to (.yml) config file."
    )
    # TODO read from given checkpoint
    # parser.add_argument(
    #     "--load-checkpoint",
    #     type=str,
    #     default="",
    #     help="Path to load saved checkpoint from."
    # )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as fp:
        cfg_dict = yaml.load(fp) #, Loader=yaml.FullLoader)
        cfg = Config(cfg_dict)

    cfg.configuration_path = configargs.config
    return cfg


if __name__ == "__main__":
    # Debugging
    config = getcfg()
    if config.abs is None:
        print("None")
    print(config)
