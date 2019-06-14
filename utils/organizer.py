import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import os


# Create test images from interpolation to test metrics benchmarking
def image_saver(obj, f_name):
    path_gt = 'benchmark/out_img/'
    if f_name.endswith('/'):
        f_name = f_name[:-1]
    if not os.path.exists(f'{path_gt}{f_name}'):
        try:
            os.makedirs(f'{path_gt}{f_name}')
            os.makedirs(f'{path_gt}{f_name}/Intensity')
            os.makedirs(f'{path_gt}{f_name}/Stokes')
        except FileExistsError:
            # directory already exists
            pass

    names = ['i0.png', 'i45.png', 'i90.png', 'i135.png']

    if not os.path.exists(f'{path_gt}{f_name}/Intensity'):
        try:
            os.makedirs(f'{path_gt}{f_name}/Intensity')
        except FileExistsError:
            # directory already exists
            pass

    for i in range(obj.images.shape[0]):
        imsave(f'{path_gt}{f_name}/Intensity/{names[i]}',
               np.array(obj.images[i, :, :], dtype=np.float))

    imsave(f'{path_gt}{f_name}/rgb_aop.png', obj.rgb_aop())

    imsave(f'{path_gt}{f_name}/rgb_pola.png', obj.rgb_pola())

    names = ['s0.png', 's1.png', 's2.png']

    if not os.path.exists(f'{path_gt}{f_name}/Stokes'):
        try:
            os.makedirs(f'{path_gt}{f_name}/Stokes')
        except FileExistsError:
            # directory already exists
            pass

    for i in range(obj.stokes.shape[0]):
        imsave(
            f'{path_gt}{f_name}/Stokes/{names[i]}',
            obj.stokes[i, :, :])

    imsave(f'{path_gt}{f_name}/I.png', obj.inte)
    imsave(f'{path_gt}{f_name}/AoP.png', obj.aop)
    imsave(f'{path_gt}{f_name}/DoP.png', obj.dop)
