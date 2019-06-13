from .bicubicSpline import *
import numpy as np
from scipy.signal import convolve2d
import math


# Image interpolation for division of focal plane polarimeters
# with intensity correlation
def intensity_correlation(raw):
    (rows, cols) = raw.shape
    img_inv = np.zeros(rows, cols)
    img_errorh = np.zeros(rows, cols)
    img_errorv = np.zeros(rows, cols)

    H = np.array([[1, 0, 1], [0, 0, 0], [-1, 0, -1]])
    V = np.array([[1, 0, -1], [0, 0, 0], [1, 0, -1]])

    img_errorh = abs(convolve2d(raw, H, boundary='symm', mode='same'))
    img_errorv = abs(convolve2d(raw, V, boundary='symm', mode='same'))

    img_errorh[0, :] = 0
    img_errorh[-1, :] = 0
    img_errorh[:, 0] = 0
    img_errorh[:, -1] = 0
    img_errorv[0, :] = 0
    img_errorv[-1, :] = 0
    img_errorv[:, 0] = 0
    img_errorv[:, -1] = 0

    R = bicubic_spline(np.array(raw, dtype=np.double))

    img0 = R[:, :, 0]
    img45 = R[:, :, 1]
    img90 = R[:, :, 2]
    img135 = R[:, :, 3]

    TB = np.array([[1 / 4, 0, 1 / 4], [0, 0, 0], [1 / 4, 0, 1 / 4]])
    img_inv = convolve2d(raw, TB, mode='same')
    img_inv[0, :] = img_inv[2, :]
    img_inv[-1, :] = img_inv[-3, :]
    img_inv[:, 0] = img_inv[:, 2]
    img_inv[:, -1] = img_inv[:, -3]

    # Diagonal Direction Interpolation
    aa = 1

    for i in range(3, rows - 2):
        for j in range(3, cols - 2):
            d1 = abs(img[i - 1, j + 1] - img[i - 3, j + 3]) + abs(img[i + 1, j - 1] -
                                                                  img[i - 1, j + 1]) + abs(img[i + 3, j - 3] - img[i + 1, j - 1]) + ...
            abs(img[i - 1, j - 1] - img[i - 3, j + 1]) + abs(img[i + 1, j - 3] -
                                                             img[i - 1, j - 1]) + abs(img[i - 1, j - 3] - img[i - 3, j - 1]) + ...
            abs(img[i + 1, j + 1] - img[i - 1, j + 3]) + abs(img[i + 3, j - 1] -
                                                             img[i + 1, j + 1]) + abs(img[i + 3, j + 1] - img[i + 1, j + 3]) + ...
            abs(img[i + 2, j - 2] + img[i - 2, j + 2] - 2 * img[i, j])
            d2 = abs(img[i - 1, j - 1] - img[i - 3, j - 3]) + abs(img[i + 1, j + 1] -
                                                                  img[i - 1, j - 1]) + abs(img[i + 3, j + 3] - img[i + 1, j + 1]) + ...
            abs(img[i - 3, j - 1] - img[i - 1, j + 1]) + abs(img[i + 1, j + 3] -
                                                             img[i - 1, j + 1]) + abs(img[i - 3, j + 1] - img[i - 1, j + 3]) + ...
            abs(img[i - 1, j - 3] - img[i + 1, j - 1]) + abs(img[i + 3, j + 1] -
                                                             img[i + 1, j - 1]) + abs(img[i + 3, j - 1] - img[i + 1, j - 3]) + ...
            abs(img[i - 2, j - 2] + img[i + 2, j + 2] - 2 * img[i, j])

            if d1 > aa * d2:
                img_inv[i, j] = -(img[i - 3, j - 3] + img[i + 3, j + 3]) / \
                    16 + (img[i - 1, j - 1] + img[i + 1, j + 1]) * 9 / 16
            elif aa * d1 < d2:
                img_inv[i, j] = -(img[i - 3, j + 3] + img[i + 3, j - 3]) / \
                    16 + (img[i - 1, j + 1] + img[i + 1, j - 1]) * 9 / 16
