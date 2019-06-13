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


#cubic spline


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



# Function for convolution bicubic interpolation
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#padding function for convolution bicubic interpolation
def padding(img,H,W):
    zimg = np.zeros((H+4,W+4))
    zimg[2:H+2,2:W+2] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2]=img[:,0:1]
    zimg[H+2:H+4,2:W+2]=img[H-1:H,:]
    zimg[2:H+2,W+2:W+4]=img[:,W-1:W]
    zimg[0:2,2:W+2]=img[0:1,:]
    #Pad the missing eight points
    zimg[0:2,0:2]=img[0,0]
    zimg[H+2:H+4,0:2]=img[H-1,0]
    zimg[H+2:H+4,W+2:W+4]=img[H-1,W-1]
    zimg[0:2,W+2:W+4]=img[0,W-1]
    return zimg

# Bicubic interpolation
def bicubic(img, ratio, a):
    #Get image size
    H,W = img.shape

    img = padding(img,H,W)
    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW))

    h = 1/ratio

    inc = 0

    for j in range(dH):
        for i in range(dW):
            x, y = i * h + 2 , j * h + 2

            x1 = 1 + x - math.floor(x)
            x2 = x - math.floor(x)
            x3 = math.floor(x) + 1 - x
            x4 = math.floor(x) + 2 - x

            y1 = 1 + y - math.floor(y)
            y2 = y - math.floor(y)
            y3 = math.floor(y) + 1 - y
            y4 = math.floor(y) + 2 - y

            mat_l = np.matrix([[u(x1,a**1),u(x2,a**2),u(x3,a**3),u(x4,a**4)]])
            mat_m = np.matrix([[img[int(y-y1),int(x-x1)],img[int(y-y2),int(x-x1)],img[int(y+y3),int(x-x1)],img[int(y+y4),int(x-x1)]],
                               [img[int(y-y1),int(x-x2)],img[int(y-y2),int(x-x2)],img[int(y+y3),int(x-x2)],img[int(y+y4),int(x-x2)]],
                               [img[int(y-y1),int(x+x3)],img[int(y-y2),int(x+x3)],img[int(y+y3),int(x+x3)],img[int(y+y4),int(x+x3)]],
                               [img[int(y-y1),int(x+x4)],img[int(y-y2),int(x+x4)],img[int(y+y3),int(x+x4)],img[int(y+y4),int(x+x4)]]])

            # mat_m = np.matrix([[0*img[int(y-y1),int(x-x1)],2*img[int(y-y2),int(x-x1)],0*img[int(y+y3),int(x-x1)],0*img[int(y+y4),int(x-x1)]],
            #                    [-1*img[int(y-y1),int(x-x2)],0*img[int(y-y2),int(x-x2)],1*img[int(y+y3),int(x-x2)],0*img[int(y+y4),int(x-x2)]],
            #                    [2*img[int(y-y1),int(x+x3)],-5*img[int(y-y2),int(x+x3)],4*img[int(y+y3),int(x+x3)],-1*img[int(y+y4),int(x+x3)]],
            #                    [-1*img[int(y-y1),int(x+x4)],3*img[int(y-y2),int(x+x4)],-3*img[int(y+y3),int(x+x4)],1*img[int(y+y4),int(x+x4)]]])

            mat_r = np.matrix([[u(y1,a**1)],[u(y2,a**2)],[u(y3,a**3)],[u(y4,a**4)]])
            dst[j, i] = np.dot(np.dot(mat_l, mat_m),mat_r)
    return dst

# -- Path of the flatfield file
#FLATF = np.genfromtxt('flatfield.csv',
#                      dtype=float, delimiter=',', skip_header=9)
#
#FLATF[FLATF >= 1.4] = 1.4  # Remove outliers or deadpixels


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
        chaine = '+-----------+\n|{p0:^5}|{p1:^5}|\n|-----+-----|\n|{p2:^5}|{p3:^5}|\n+-----------+'.format(**dico)
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
#        self.raw = np.asarray((raw * FLATF) / FLATF.max(), dtype=self.raw.dtype)

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

        self.error = sum((images - np.tensordot(np.dot(imat, mat), images, 1))**2, 0)


    @property
    def inte(self):
        """Return intensity image"""
        return self.stokes[0]


    @property
    def aop(self):
        """Return aop image"""
        return np.mod(np.arctan2(self.stokes[2], self.stokes[1])/ 2., np.pi)


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
        aop_rgb = cmap(np.mod(newaop, np.pi)/np.pi)

        if opencv:  # opencv compatibilty
            return np.uint8(aop_rgb[:, :, [2, 1, 0]]*255)

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

        hsv = np.zeros(self.aop.shape+(3, ))
        # -- Normalization in [0. 1.]
        hsv[:, :, 2] = self.inte / 2. / (2**self.depth - 1)
        hsv[:, :, 1] = np.minimum(self.dop / dop_max, 1)
        hsv[:, :, 0] = self.aop / np.pi

        hsv[self.dop <= dop_min, :] = (0., 0., 0.)

        #to be checked
        rgb = hsv_to_rgb(hsv)
        if opencv:  # opencv compatibilty
            return np.uint8(rgb[:, :, [2, 1, 0]]*255)
        return rgb


# %% Functions


def raw2quad(raw, method='none', pixels_order=Pixorder.polarcamV2):
    r"""Convert a raw image from polarcam into a list of ordered images
    [I0, I45, I90, I135]. If the parameter `method` is set to none the output images will have
    half size of the original image.

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
                   np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])/2.,
                   np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])/4.,
                   np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])/2.]
    elif method == 'conv_bicubic':
        I = np.array(raw,dtype=np.double)

        I0 = I[0::2, 0::2]
        I45 = I[0::2, 1::2]
        I90 = I[1::2, 1::2]
        I135 = I[1::2, 0::2]
        I0 = bicubic(I0, 2, 1/2)
        I45 = bicubic(I45, 2, 1/2)
        I90 = bicubic(I90, 2, 1/2)
        I135 = bicubic(I135, 2, 1/2)

        R = np.dstack((I0,I45,I90,I135))
        images = np.zeros((4,I0.shape[0],I0.shape[1]))

        for i in range(4):
            images[i,:,:] = np.array(R[:,:,i],dtype=np.double)

        return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)


    elif method == 'weightedb3':
        b = np.sqrt(2)/2/(np.sqrt(2)/2+np.sqrt(10))
        a = np.sqrt(10)/2/(np.sqrt(2)/2+np.sqrt(10))

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
        c = np.sqrt(2)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))
        b = np.sqrt(10)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))
        a = 3*np.sqrt(2)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))

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
        I = np.array(raw,dtype=np.double)

        (m, n) = I.shape

        R = np.zeros((m, n, 4))

        # Labeling different polatiration channels
        O = np.zeros((m, n), dtype=int)

        step = 1
        O[0::2, 0::2] = 0
        O[0::2, 1::2] = 1
        O[1::2, 1::2] = 2
        O[1::2, 0::2] = 3


        # Store intermediate results
        Y1 = np.array(raw,dtype=np.double)
        Y2 = np.array(raw,dtype=np.double)
        #Y1 = np.double(I)
        #Y2 = np.double(I)Y1 - I

        # for index in range(R.shape[2]):
        R[:, :, 0] = np.array(raw,dtype=np.double)
        R[:, :, 1] = np.array(raw,dtype=np.double)
        R[:, :, 2] = np.array(raw,dtype=np.double)
        R[:, :, 3] = np.array(raw,dtype=np.double)

        '''
        % Stage one interpolation: interpolate vertically for case Fig.6(b),
        % interpolate horizontally for case Fig.6(c), interpolate in diagonal
        % directions for case Fig.6(a). The Eqs.(14)-(17) are simplified in this
        % code.
        '''

        for i in range(3, m-3):
            for j in range(3, n-3):
                R[i, j, O[i, j]] = I[i, j]
                R[i, j, O[i, j+1]] = 0.5*I[i, j] + 0.0625*I[i, j-3] - 0.25*I[i, j-2] + \
                    0.4375*I[i, j-1] + 0.4375*I[i, j+1] - \
                    0.25*I[i, j+2] + 0.0625*I[i, j+3]
                R[i, j, O[i+1, j]] = 0.5*I[i, j] + 0.0625*I[i-3, j] - 0.25*I[i-2, j] + \
                    0.4375*I[i-1, j] + 0.4375*I[i+1, j] - \
                    0.25*I[i+2, j] + 0.0625*I[i+3, j]
                Y1[i, j] = 0.5*I[i, j] + 0.0625*I[i-3, j-3] - 0.25*I[i-2, j-2] + 0.4375 * \
                    I[i-1, j-1] + 0.4375*I[i+1, j+1] - \
                    0.25*I[i+2, j+2] + 0.0625*I[i+3, j+3]
                Y2[i, j] = 0.5*I[i, j] + 0.0625*I[i-3, j+3] - 0.25*I[i-2, j+2] + 0.4375 * \
                    I[i-1, j+1] + 0.4375*I[i+1, j-1] - \
                    0.25*I[i+2, j-2] + 0.0625*I[i+3, j-3]
        # One can adjust for better result
        thao = 2.8
        # Fusion of the estimations with edge classifier for case Fig.6(a).

        for i in range(3, m-3):
            for j in range(3, n-3):
                pha1 = 0.0
                pha2 = 0.0

                for k in range(-2, 3, 2):
                    for l in range(-2, 3, 2):
                        pha1 = pha1 + abs(Y1[i+k, j+l] - I[i+k, j+l])
                        pha2 = pha2 + abs(Y2[i+k, j+l] - I[i+k, j+l])

                if (pha1 / pha2) > thao:
                    R[i, j, O[i+1, j+1]] = Y2[i, j]
                elif (pha2/pha1) > thao:
                    R[i, j, O[i+1, j+1]] = Y1[i, j]
                elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                    d1 = abs(I[i-1, j-1] - I[i+1, j+1]) + \
                        abs(2*I[i, j] - I[i-2, j-2] - I[i+2, j+2])
                    d2 = abs(I[i+1, j-1] - I[i-1, j+1]) + \
                        abs(2*I[i, j] - I[i+2, j-2] - I[i-2, j+2])
                    epsl = 0.000000000000001
                    w1 = 1/(d1 + epsl)
                    w2 = 1/(d2+epsl)
                    R[i, j, O[i+1, j+1]] = (w1*Y1[i, j] + w2*Y2[i, j])/(w1 + w2)

        RR = np.array(R,dtype=np.double)

        XX1 = np.array(raw,dtype=np.double)
        XX2 = np.array(raw,dtype=np.double)
        YY1 = np.array(raw,dtype=np.double)
        YY2 = np.array(raw,dtype=np.double)

        # Stage two interpolation: interpolate horizontally for case Fig.6(b),
        # interpolate vertically for case Fig.6(c).

        for i in range(3, m-3):
            for j in range(3, n-3):
                XX1[i, j] = R[i, j, O[i, j+1]]
                XX2[i, j] = 0.5*I[i, j] + 0.0625 * \
                    R[i-3, j, O[i, j+1]] - 0.25*I[i-2, j]
                XX2[i, j] = XX2[i, j] + 0.4375 * \
                    R[i-1, j, O[i, j+1]] + 0.4375*R[i+1, j, O[i, j+1]]
                XX2[i, j] = XX2[i, j] - 0.25*I[i+2, j] + 0.0625*R[i+3, j, O[i, j+1]]
                YY1[i, j] = R[i, j, O[i+1, j]]
                YY2[i, j] = 0.5*I[i, j] + 0.0625 * \
                    R[i, j-3, O[i+1, j]] - 0.25*I[i, j-2]
                YY2[i, j] = YY2[i, j] + 0.4375 * \
                    R[i, j-1, O[i+1, j]] + 0.4375*R[i, j+1, O[i+1, j]]
                YY2[i, j] = YY2[i, j] - 0.25*I[i, j+2] + 0.0625*R[i, j+3, O[i+1, j]]

        # Fusion of the estimations with edge classifier for case Fig.6(b) and Fig.6(c).

        for i in range(3, m-4):
            for j in range(3, n-4):
                pha1 = 0.0
                pha2 = 0.0

                for k in range(-2, 3, 2):
                    for l in range(-2, 3, 2):
                        pha1 = pha1 + abs(XX1[i+k, j+l] - I[i+k, j+l])
                        pha2 = pha2 + abs(XX2[i+k, j+l] - I[i+k, j+l])

                if (pha1 / pha2) > thao:
                    R[i, j, O[i, j+1]] = XX2[i, j]
                elif (pha2/pha1) > thao:
                    R[i, j, O[i, j+1]] = XX1[i, j]
                elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                    d1 = abs(I[i, j-1] - I[i, j+1]) + \
                        abs(2*I[i, j] - I[i, j-2] - I[i, j+2])
                    d2 = abs(I[i+1, j] - I[i-1, j]) + \
                        abs(2*I[i, j] - I[i+2, j] - I[i-2, j])
                    epsl = 0.000000000000001
                    w1 = 1/(d1 + epsl)
                    w2 = 1/(d2 + epsl)
                    R[i, j, O[i, j+1]] = (w1*XX1[i, j] + w2*XX2[i, j])/(w1 + w2)

                pha1 = 0.0
                pha2 = 0.0

                for k in range(-2, 3, 2):
                    for l in range(-2, 3, 2):
                        pha1 = pha1 + abs(YY1[i+k, j+l] - I[i+k, j+l])
                        pha2 = pha2 + abs(YY2[i+k, j+l] - I[i+k, j+l])

                if (pha1 / pha2) > thao:
                    R[i, j, O[i+1, j]] = YY2[i, j]
                elif (pha2/pha1) > thao:
                    R[i, j, O[i+1, j]] = YY1[i, j]
                elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                    d1 = abs(I[i, j-1] - I[i, j+1]) + \
                        abs(2*I[i, j] - I[i, j-2] - I[i, j+2])
                    d2 = abs(I[i+1, j] - I[i-1, j]) + \
                        abs(2*I[i, j] - I[i+2, j] - I[i-2, j])
                    epsl = 0.000000000000001
                    w1 = 1/(d1 + epsl)
                    w2 = 1/(d2 + epsl)
                    R[i, j, O[i, j+1]] = (w1*YY1[i, j] + w2*YY2[i, j])/(w1 + w2)

        R = np.array(RR,dtype=np.double)
        I0 = np.array(R[:, :, 0],dtype=np.double)
        I45 = np.array(R[:, :, 1],dtype=np.double)
        I90 = np.array(R[:, :, 2],dtype=np.double)
        I135 = np.array(R[:, :, 3],dtype=np.double)

        images = np.zeros((4,I0.shape[0],I0.shape[1]))

        for i in range(4):
            images[i,:,:] = np.array(R[:,:,i],dtype=np.double)

        return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)

    elif method == 'bicubic_spline':

        R = bicubic_spline(np.array(raw,dtype=np.double))

        images = np.zeros((4,R.shape[0],R.shape[1]))

        for i in range(4):
            images[i,:,:] = np.array(R[:,:,i],dtype=np.double)

        return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)


    convs = [convolve2d(raw, k, mode='same') for k in kernels]
    offsets = [[(0, 0), (0, 1), (1, 1), (1, 0)],
               [(0, 1), (0, 0), (1, 0), (1, 1)],
               [(1, 1), (1, 0), (0, 0), (0, 1)],
               [(1, 0), (1, 1), (0, 1), (0, 0)]]

    images = np.zeros((4,)+raw.shape)
    for (j, o) in enumerate(offsets):
        for ide in range(4):
            images[j, o[ide][0]::2, o[ide][1]::2] = convs[ide][o[ide][0]::2, o[ide][1]::2]

    return np.asarray(images[pixels_order.value, :, :], dtype=raw.dtype)



# %% Main
if __name__ == '__main__':
    POLA = Polaim('images/image_00001.tiff',method='bicubic_spline')
    pl.imshow(POLA.rgb_aop(dop_min=0))
    pl.show()
    pl.imshow(POLA.rgb_pola(dop_max=0.4, dop_min=0))
