from polarcam import *
from utils import *
from lib import *

if __name__ == '__main__':

    test_images_create = True

    image_name = 'image_00001'
    image_path = 'images/image_00001.tiff'
    path_gt = 'benchmark/groundtruth/'

    if test_images:
        POLA = Polaim(image_path, method='bilinear')
        test_images.create_test_img(POLA)

    methods = ['linear', 'bilinear', 'weightedb3', 'weightedb4',
               'conv_bicubic', 'bicubic_spline', 'newton',
               'intensity_correlation']

    save_InterPol = True

    if save_InterPol:
        if not os.path.exists(f'{path_gt}{image_name}'):
            try:
                os.makedirs(f'{path_gt}{image_name}')
            except FileExistsError:
                # directory already exists
                pass

    obj_storage = []

    for met in methods:
        obj = Polaim(image_path, method=met)
        obj_storage.append(obj)
        if save_InterPol:
            organizer.image_saver(obj, f'{image_name}/{met}')
