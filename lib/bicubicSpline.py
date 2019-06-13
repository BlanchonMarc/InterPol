from os.path import join, exists, split
from os import makedirs
import argparse
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def bicubic_spline(raw):
    #raw = plt.imread(raw)
    (m, n) = raw.shape

    img1 = np.zeros((m, n))
    img2 = np.zeros((m, n))
    img3 = np.zeros((m, n))
    img4 = np.zeros((m, n))

    img1[0::2, 0::2] = raw[0::2, 0::2]
    img2[0::2, 1::2] = raw[0::2, 1::2]
    img3[1::2, 1::2] = raw[1::2, 1::2]
    img4[1::2, 0::2] = raw[1::2, 0::2]

    # imag1
    x = np.arange(0, n, 2)
    y = img1[0::2, 0::2]

    pp = CubicSpline(x, np.transpose(y))

    img1[0::2, 0:n - 1] = np.transpose(pp(np.arange(0, n - 1)))

    x1 = np.arange(0, m, 2)
    y1 = img1[0::2, 0:n - 1]

    pp1 = CubicSpline(x1, y1)

    img1[0:m-1, 0:n - 1] = pp1(np.arange(0, m - 1))

    img1[m-1, :] = img1[m - 2, :]
    img1[:, n-1] = img1[:, n - 2]


    # img2

    x = np.arange(1, n, 2)
    y = img2[0::2, 1::2]

    pp = CubicSpline(x, np.transpose(y))

    img2[0::2, 1:n] = np.transpose(pp(np.arange(1, n)))

    x1 = np.arange(0, m, 2)
    y1 = img2[0::2, 1:n]

    pp1 = CubicSpline(x1, y1)

    img2[0:m-1, 1:n] = pp1(np.arange(0, m - 1))

    img2[m-1, :] = img2[m-2, :]
    img2[:, 0] = img2[:, 1]


    # img3

    x = np.arange(1, n, 2)
    y = img3[1::2, 1::2]

    pp = CubicSpline(x, np.transpose(y))

    img3[1::2, 1:n] = np.transpose(pp(np.arange(1, n)))

    x1 = np.arange(1, m, 2)
    y1 = img3[1::2, 1:n]

    pp1 = CubicSpline(x1, y1)

    img3[1:m, 1:n] = pp1(np.arange(1, m ))

    img3[0, :] = img3[1, :]
    img3[:, 0] = img3[:, 1]


    # img4

    x = np.arange(0, n, 2)
    y = img4[1::2, 0::2]

    pp = CubicSpline(x, np.transpose(y))

    img4[1::2, 0:n-1] = np.transpose(pp(np.arange(0, n-1)))

    x1 = np.arange(1, m, 2)
    y1 = img3[1::2, 0:n]

    pp1 = CubicSpline(x1, y1)

    img4[1:m, 0:n] = pp1(np.arange(1, m ))

    img4[0, :] = img4[1, :]
    img4[:, n-1] = img4[:, n-2]


    return np.dstack((img1, img2, img3, img4))
    # print(img4.shape)
    # plt.imshow(img4)
    # plt.show()


# test
if __name__ == '__main__':
    cubic_spline('/Users/marc/Github/InterPol/images/image_00001.tiff')
