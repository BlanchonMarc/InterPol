from polarcam import *
from utils import *
from lib import *
import pandas as pd


if __name__ == '__main__':
    with np.errstate(divide='ignore', invalid='ignore'):
        test_images_create = False

        image_name = 'image_00001'
        image_path = 'images/image_00001.tiff'
        path_gt = 'benchmark/out_img/'

        gt = Polaim(image_path, method='bilinear')

        if test_images:
            test_images.create_test_img(gt)

        methods = ['linear', 'bilinear', 'weightedb3', 'weightedb4',
                   'conv_bicubic', 'bicubic_spline', 'newton',
                   'intensity_correlation']

        save_InterPol = False

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

        df_list, assoc_names = metrics.evaluate(gt, obj_storage)

        print(assoc_names[0])
        print(df_list[0])

        path_df = '/Users/marc/Github/InterPol/benchmark/data_frames/'

        if not os.path.exists(f'{path_df}{image_name}'):
            try:
                os.makedirs(f'{path_df}{image_name}')
            except FileExistsError:
                # directory already exists
                pass

        for idx, t_df in enumerate(df_list):
            t_df.to_csv(f'{path_df}{image_name}/{assoc_names[idx]}.csv')
