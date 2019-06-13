import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave


# Create test images from interpolation to test metrics benchmarking
def create_test_img(obj):
    names = ['i0.png', 'i45.png', 'i90.png', 'i135.png']
    for i in range(obj.images.shape[0]):
        imsave(f'benchmark/groundtruth/test/Intensity/{names[i]}',
               np.array(obj.images[i, :, :], dtype=np.float))

    imsave(f'benchmark/groundtruth/test/rgb_aop.png', obj.rgb_aop())

    imsave(f'benchmark/groundtruth/test/rgb_pola.png', obj.rgb_pola())

    names = ['s0.png', 's1.png', 's2.png']
    for i in range(obj.stokes.shape[0]):
        imsave(
            f'benchmark/groundtruth/test/Stokes/{names[i]}',
            obj.stokes[i, :, :])

    imsave(f'benchmark/groundtruth/test/I.png', obj.inte)
    imsave(f'benchmark/groundtruth/test/AoP.png', obj.aop)
    imsave(f'benchmark/groundtruth/test/DoP.png', obj.dop)
