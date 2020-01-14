from polarcam import *
from utils import *
from lib import *
import pandas as pd
import matplotlib.pyplot as pl
import sys
from scipy.misc import imsave

if __name__ == '__main__':
    with np.errstate(divide='ignore', invalid='ignore'):
        test_images_create = False

        image_name = 'Carte'
        # image_path = 'images/image_00001.tiff'
        paths_gt = ['images/Expérimentations_Pola/Carte_0.bmp',
                    'images/Expérimentations_Pola/Carte_45.bmp',
                    'images/Expérimentations_Pola/Carte_90.bmp',
                    'images/Expérimentations_Pola/Carte_135.bmp']
        path_gt = 'benchmark/out_img/'

        gt = PolaGT(paths_gt[0], paths_gt[1], paths_gt[2], paths_gt[3])

        gt_p0 = pl.imread(paths_gt[0])

        image_created = np.zeros_like(gt_p0)

        gt_p0 = gt_p0[0::2, 0::2]
        gt_p45 = pl.imread(paths_gt[1])
        gt_p45 = gt_p45[0::2, 1::2]
        gt_p90 = pl.imread(paths_gt[2])
        gt_p90 = gt_p90[1::2, 1::2]
        gt_p135 = pl.imread(paths_gt[3])
        gt_p135 = gt_p135[1::2, 0::2]

        image_created[0::2, 0::2] = gt_p90
        image_created[0::2, 1::2] = gt_p135
        image_created[1::2, 1::2] = gt_p0
        image_created[1::2, 0::2] = gt_p45

        image_path = 'images/Carte_DOFP.bmp'

        imsave(image_path, image_created)

        if test_images_create:
            test_images.create_test_img(gt)

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
