from dataio import blender


def load_data(cfg):
    if cfg.dataset.type == 'blender':
        return blender.load(cfg)

    if cfg.dataset.type == 'llff':
        # TODO
        print("LLFF data not supported yet")
        quit(1)

    if cfg.dataset.type == 'deepvoxels':
        # TODO
        print("Deepvoxel data not supported yet")
        quit(1)

    print('Unknown dataset type', cfg.dataset.type, 'exiting')
    quit(1)
