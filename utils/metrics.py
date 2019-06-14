from scipy.ndimage import generic_laplace, uniform_filter, correlate
from scipy.ndimage import gaussian_filter
from scipy import misc
from scipy import signal
import numpy as np
import math
import cv2

# imeak Signal to Noise Ratio


def imSNR(gt, im):
    mse = np.mean((gt - im) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


# Cross Correlation between two 2D image
def Cross_Corr_2D(gt, im):
    return signal.correlate2d(im, gt, boundary='symm', mode='same')


# Mutual information
# I(X;Y) = \sum_{y \in Y} \sum_{x \in X} im(x,y) \log{ \left(\frac{im(x,y)}{im(x)\,im(y)}\right) }
def mutual_information(gt, im, bins=255):
    hist_2d, x_edges, y_edges = np.histogram2d(gt.ravel(),
                                               im.ravel(),
                                               bins=bins)
    imxy = hgram / float(nim.sum(hgram))
    imx = nim.sum(imxy, axis=1)
    imy = nim.sum(imxy, axis=0)
    imx_imy = imx[:, None] * imy[None, :]
    nzs = imxy > 0
    return np.sum(imxy[nzs] * np.log(imxy[nzs] / imx_imy[nzs]))


# Mean Square error
def MSE(gt, im):
    return np.mean((gt.astyime(np.float64) - im.astyime(np.float64))**2)


# Root Mean Square Error
def RMSE(gt, im):
    return np.sqrt(MSE(gt, im))


# UQI - Universal Quality Index
def UQI(gt, im, s=8):
    N = s**2
    window = np.ones((s, s))

    gt_sq = gt * gt
    im_sq = im * im
    gt_im = gt * im

    gt_sum = uniform_filter(gt, s)
    im_sum = uniform_filter(im, s)
    gt_sq_sum = uniform_filter(gt_sq, s)
    im_sq_sum = uniform_filter(im_sq, s)
    gt_im_sum = uniform_filter(gt_im, s)

    gt_im_sum_mul = gt_sum * im_sum
    gt_im_sum_sq_sum_mul = gt_sum * gt_sum + im_sum * im_sum
    numerator = 4 * (N * gt_im_sum - gt_im_sum_mul) * gt_im_sum_mul
    denominator1 = N * (gt_sq_sum + im_sq_sum) - gt_im_sum_sq_sum_mul
    denominator = denominator1 * gt_im_sum_sq_sum_mul

    q_maim = np.ones(denominator.shaime)
    index = np.logical_and((denominator1 == 0), (gt_im_sum_sq_sum_mul != 0))
    q_maim[index] = 2 * gt_im_sum_mul[index] / gt_im_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_maim[index] = numerator[index] / denominator[index]

    s = int(np.round(s / 2))
    return np.mean(q_maim[s:-s, s:-s])


class Filter(Enum):
    UNIFORM = 0
    GAUSSIAN = 1


def filter2(img, fltr, mode='same'):
    return signal.convolve2d(img, np.rot90(fltr, 2), mode=mode)


def _get_sums(gt, im, win, mode='same'):
    mu1, mu2 = (filter2(gt, win, mode), filter2(im, win, mode))
    return mu1 * mu1, mu2 * mu2, mu1 * mu2


def _get_sigmas(gt, im, win, mode='same', **kwargs):
    if 'sums' in kwargs:
        gt_sum_sq, im_sum_sq, gt_im_sum_mul = kwargs['sums']
    else:
        gt_sum_sq, im_sum_sq, gt_im_sum_mul = _get_sums(gt, im, win, mode)

    return filter2(gt * gt, win, mode) - gt_sum_sq,\
        filter2(im * im, win, mode) - im_sum_sq, \
        filter2(gt * im, win, mode) - gt_im_sum_mul


# fspecial function
def fspecial(fltr, ws, **kwargs):
    if fltr == Filter.UNIFORM:
        return np.ones((ws, ws)) / ws**2
    elif fltr == Filter.GAUSSIAN:
        x, y = np.mgrid[-ws // 2 + 1:ws // 2 + 1, -ws // 2 + 1:ws // 2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0 * kwargs['sigma']**2)))
        g[g < np.finfo(g.dtype).eps * g.max()] = 0
        assert g.shape == (ws, ws)
        den = g.sum()
        if den != 0:
            g /= den
        return g
    return None


# Structural Similarity Index SSIM
def ssim(gt, im, ws=11, C1=0.01, C2=0.03, MAX=None,
         fltr_specs=None, mode='valid'):

    if MAX is None:
        MAX = np.iinfo(gt.dtype).max

    if fltr_specs is None:
        fltr_specs = dict(fltr=Filter.UNIFORM, ws=ws)

    win = fspecial(fltr_specs)

    gt_sum_sq, im_sum_sq, gt_im_sum_mul = _get_sums(gt, im, win, mode)
    sigmagt_sq, sigmaim_sq, sigmagt_im = _get_sigmas(
        gt, im, win, mode, sums=(gt_sum_sq, im_sum_sq, gt_im_sum_mul))

    assert C1 > 0
    assert C2 > 0

    ssim_map = ((2 * gt_im_sum_mul + C1) * (2 * sigmagt_im + C2)) / \
        ((gt_sum_sq + im_sum_sq + C1) * (sigmagt_sq + sigmaim_sq + C2))
    cs_map = (2 * sigmagt_im + C2) / (sigmagt_sq + sigmaim_sq + C2)

    return np.mean(ssim_map), np.mean(cs_map)
