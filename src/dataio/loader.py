from input_data import blender


def load_blender_data(args):
    images, poses, render_poses, hwf, i_split = blender.load(
        args.datadir, args.half_res, args.testskip)

    print('Loaded blender', images.shape, poses.shape,
          render_poses.shape, hwf, args.datadir)

    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]
    return images, poses, render_poses, i_train, i_val, i_test, near, far, hwf


def load_data(args):
    if args.dataset_type == 'blender':
        return load_blender_data(args)

    if args.dataset_type == 'llff':
        # TODO
        print("LLFF data not supported yet")
        quit(1)

    if args.dataset_type == 'deepvoxels':
        # TODO
        print("Deepvoxel data not supported yet")
        quit(1)

    print('Unknown dataset type', args.dataset_type, 'exiting')
    quit(1)
