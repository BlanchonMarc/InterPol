#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:20:00 2018

@authors: olivierm, BlanchonMarc

Tools to read and import data from Polarcam
"""

from enum import Enum
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as pl
from matplotlib.colors import hsv_to_rgb
import math
from scipy.interpolate import CubicSpline
from lib import *


# -- Path of the flatfield file
# FLATF = np.genfromtxt('flatfield.csv',
#                      dtype=float, delimiter=',', skip_header=9)
#
# FLATF[FLATF >= 1.4] = 1.4  # Remove outliers or deadpixels


# %% Pixel Order

# positions are defined clockwise
# +-------+
# | 0 | 1 |
# |---+---|
# | 3 | 2 |
# +-------+


# by default I45_90_135_0 for polarcam V1
# +-----------+
# | I45|  I90 |
# |----+------|
# | I0 | I135 |
# +-----------+


# by default I90_135_0_45 for polarcam V2
# +-----------+
# | I90| I135 |
# |----+------|
# | I45|  I0  |
# +-----------+

class Pixorder(Enum):
    """ Class that defines order of pixels """
    polarcamV2 = (2, 3, 0, 1)
    polarcamV1 = (3, 0, 1, 2)

    I0_45_90_135 = (0, 1, 2, 3)
    I45_0_135_90 = (1, 0, 3, 2)
    I135_90_45_0 = (3, 2, 1, 0)
    I90_135_0_45 = (2, 3, 0, 1)
    I45_90_135_0 = (3, 0, 1, 2)

    def __str__(self):
        posi = ['p{}'.format(u) for u in list(self.value)]
        filt = ('I0', 'I45', 'I90', 'I135')
        dico = dict(zip(posi, filt))
        chaine = '+-----------+\n|{p0:^5}|{p1:^5}|\n|-----+-----|\n|{p2:^5}|{p3:^5}|\n+-----------+'.format(
            **dico)
        return chaine

# %% Polaim


class Polaim():
    """Class that describes pixelated images -- version 2018"""

    def __init__(self, raw, method='none', pixels_order=Pixorder.polarcamV2):
        """ Initialization method """

        # --- Open the image if filename provided
        if isinstance(raw, str):
            raw = pl.imread(raw)

        self.depth = 8  # 8bit depth by default

        # --- In cas of 10 bit depth convert into the right range
        if raw.dtype == 'uint16':
            self.depth = 16

        # --- Apply the flat field correction
        # self.raw = np.asarray((raw * FLATF) / FLATF.max(), dtype=self.raw.dtype)

        # --- Extract the 4 images using interpolation technics
        images = raw2quad(raw, method=method, pixels_order=pixels_order)

        # --- Compute the 3 stokes parameters
        mat = np.array([[0.5, 0.5, 0.5, 0.5],
                        [1.0, 0.0, -1., 0.0],
                        [0.0, 1.0, 0.0, -1.]])

        self.stokes = np.tensordot(mat, images, 1)

        # --- Error estimation
        imat = 0.5 * np.array([[1, 1, 0.],
                               [1, 0, 1.],
                               [1, -1, 0],
                               [1, 0, -1]])

        self.error = sum(
            (images - np.tensordot(np.dot(imat, mat), images, 1))**2, 0)

    @property
    def inte(self):
        """Return intensity image"""
        return self.stokes[0]

    @property
    def aop(self):
        """Return aop image"""
        return np.mod(np.arctan2(self.stokes[2], self.stokes[1]) / 2., np.pi)

    @property
    def dop(self):
        """Return dop image"""
        return np.divide(np.sqrt(self.stokes[2]**2, self.stokes[1]**2),
                         self.stokes[0], out=np.zeros_like(self.stokes[0]),
                         where=self.stokes[0] != 0)

    def rgb_aop(self, colormap='hsv', dop_min=0.0, opencv=False):
        r""" Given a Polaim object return a RGB image of the aop

        Parameters
        ----------
        colormap : string
            colormap used for aop
        opencv : boolean
            assume opencv color representation convention
        aop_only : boolean
            assume dop=1 and constant intensity

        Return
        ------
        col : 3D array
            RGB image representing the aop

        Examples
        --------
        >>> imp.aop2rgb()
        """

        newaop = np.ma.array(self.aop.copy())  # convert into masked array
        newaop.mask = self.dop <= dop_min
        cmap = pl.get_cmap(colormap)
        cmap.set_bad((0., 0., 0.))
        # cmap.set_under((0, 0, 0))
        # cmap.set_over((0, 0, 0))
        aop_rgb = cmap(np.mod(newaop, np.pi) / np.pi)

        if opencv:  # opencv compatibilty
            return np.uint8(aop_rgb[:, :, [2, 1, 0]] * 255)

        return aop_rgb

    def rgb_pola(self, dop_max=1.0, dop_min=0.0, opencv=False):
        r""" Given a Polaim object return a RGB image
        with HSV mapping

        Parameters
        ----------
        dop_max : floatting number
            maximum authorized value for dop
        opencv : boolean
            assume opencv color representation convention

        Return
        ------
        col : 3D array
            RGB image representing the aop

        Examples
        --------
        >>> imp.pola2rgb()
        """

        hsv = np.zeros(self.aop.shape + (3, ))
        # -- Normalization in [0. 1.]
        hsv[:, :, 2] = self.inte / 2. / (2**self.depth - 1)
        hsv[:, :, 1] = np.minimum(self.dop / dop_max, 1)
        hsv[:, :, 0] = self.aop / np.pi

        hsv[self.dop <= dop_min, :] = (0., 0., 0.)

        # to be checked
        rgb = hsv_to_rgb(hsv)
        if opencv:  # opencv compatibilty
            return np.uint8(rgb[:, :, [2, 1, 0]] * 255)
        return rgb


# %% Functions


def raw2quad(raw, method='none', pixels_order=Pixorder.polarcamV2):
    """Convert a raw image from polarcam into a list of ordered images
    [I0, I45, I90, I135]. If the parameter `method` is set to none the
    output images will have half size of the original image.

    Parameters
    ----------
    raw : 2D array
        Original polarcam image in grayscale
    method : {'none', 'linear', 'bilinear', 'weightedb3', 'weightedb4'}
        'none' : no interpolation method is performed
        'linear' : linear interpolation
        'bilinear' : bilinear interpolation
        'weightedb3' or 'weightedb4' : weighted bilinear interpolation with 3
        or 4 neighbors

    Returns
    -------
    images : 3D array containing 4 images

    Example
    -------
    >>> images = raw2quad(raw, method='bilinear')

    """
    if method == 'none':
        images = np.array([raw[0::2, 0::2],  # 0
                           raw[0::2, 1::2],  # 1
                           raw[1::2, 1::2],  # 2
                           raw[1::2, 0::2]])  # 3
        return images[pixels_order.value, :, :]
    if method == 'linear':
        kernels = [np.array([[1, 0], [0, 0.]]),
                   np.array([[0, 1], [0, 0.]]),
                   np.array([[0, 0], [0, 1.]]),
                   np.array([[0, 0], [1, 0.]])]
    elif method == 'bilinear':
        kernels = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0.]]),
                   np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]) / 2.,
                   np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) / 4.,
                   np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]) / 2.]
    elif method == 'conv_bicubic':

        R = bicubicConv.conv_bicubic(np.array(raw, dtype=np.double))

        images = np.zeros((4, R.shape[0], R.shape[1]))

        for i in range(4):
            images[i, :, :] = np.array(R[:, :, i], dtype=np.double)

        return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)

    elif method == 'weightedb3':
        b = np.sqrt(2) / 2 / (np.sqrt(2) / 2 + np.sqrt(10))
        a = np.sqrt(10) / 2 / (np.sqrt(2) / 2 + np.sqrt(10))

        kernels = [np.array([[0, b, 0, 0],
                             [0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0]]),
                   np.array([[0, 0, b, 0],
                             [0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0],
                             [0, 0, b, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0],
                             [0, b, 0, 0]])]
    elif method == 'weightedb4':
        c = np.sqrt(2) / 2 / (3 * np.sqrt(2) / 2 +
                              np.sqrt(2) / 2 + np.sqrt(10))
        b = np.sqrt(10) / 2 / (3 * np.sqrt(2) / 2 +
                               np.sqrt(2) / 2 + np.sqrt(10))
        a = 3 * np.sqrt(2) / 2 / (3 * np.sqrt(2) / 2 +
                                  np.sqrt(2) / 2 + np.sqrt(10))

        kernels = [np.array([[0, b, 0, c],
                             [0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0]]),
                   np.array([[c, 0, b, 0],
                             [0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0],
                             [c, 0, b, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0],
                             [0, b, 0, c]])]

    elif method == 'newton':
        R = newton.newton_polynomial(np.array(raw, dtype=np.double))

        images = np.zeros((4, R.shape[0], R.shape[1]))

        for i in range(4):
            images[i, :, :] = np.array(R[:, :, i], dtype=np.double)

        return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)

    elif method == 'bicubic_spline':

        R = bicubicSpline.bicubic_spline(np.array(raw, dtype=np.double))

        images = np.zeros((4, R.shape[0], R.shape[1]))

        for i in range(4):
            images[i, :, :] = np.array(R[:, :, i], dtype=np.double)

        return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)

    else:
        raise SystemExit(f"{method} is not a method.")

    convs = [convolve2d(raw, k, mode='same') for k in kernels]
    offsets = [[(0, 0), (0, 1), (1, 1), (1, 0)],
               [(0, 1), (0, 0), (1, 0), (1, 1)],
               [(1, 1), (1, 0), (0, 0), (0, 1)],
               [(1, 0), (1, 1), (0, 1), (0, 0)]]

    images = np.zeros((4,) + raw.shape)
    for (j, o) in enumerate(offsets):
        for ide in range(4):
            images[j, o[ide][0]::2, o[ide][1]::2] = convs[ide][o[ide][0]::2, o[ide][1]::2]

    return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)


if __name__ == '__main__':
    POLA = Polaim('images/image_00001.tiff', method='conv_bicubic')
    pl.imshow(POLA.rgb_aop(dop_min=0))
    pl.show()
    pl.imshow(POLA.rgb_pola(dop_max=0.4, dop_min=0))
