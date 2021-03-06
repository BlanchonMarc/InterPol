from scipy.ndimage import generic_laplace, uniform_filter, correlate
from scipy.ndimage import gaussian_filter
from scipy import misc
from scipy import signal
import numpy as np
import math
import cv2
from enum import Enum
import sklearn.metrics as skm
import pandas as pd

# Some extraction from https://github.com/andrewekhalel/sewar/


# Peak Signal to Noise Ratio
def PSNR(gt, im):
    mse = np.mean((gt - im) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


# Cross Correlation between two 2D image
def Cross_Corr_2D(gt, im):
    return signal.correlate2d(im, gt, boundary='symm', mode='same')


# Mutual information
# I(X;Y) = \sum_{y \in Y} \sum_{x \in X} im(x,y)
# \log{ \left(\frac{im(x,y)}{im(x)\,im(y)}\right) }
def mutual_information(gt, im, bins=255):
    hist_2d, x_edges, y_edges = np.histogram2d(gt.ravel(),
                                               im.ravel(),
                                               bins=bins)
    imxy = hist_2d / float(np.sum(hist_2d))
    imx = np.sum(imxy, axis=1)
    imy = np.sum(imxy, axis=0)
    imx_imy = imx[:, None] * imy[None, :]
    nzs = imxy > 0
    return np.sum(imxy[nzs] * np.log(imxy[nzs] / imx_imy[nzs]))


# Mean Square error
def MSE(gt, im):
    return np.mean((gt.astype(np.float64) - im.astype(np.float64))**2)


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

    q_maim = np.ones(denominator.shape)
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
    return signal.convolve2d(img, np.rot90(fltr, 2, axes=(0, 1)), mode=mode)


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
def SSIM(gt, im, ws=11, C1=0.01, C2=0.03, MAX=None,
         fltr_specs=None, mode='valid'):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    if MAX is None:
        MAX = np.iinfo(gt.dtype).max

    if fltr_specs is None:
        fltr_specs = dict(fltr=Filter.UNIFORM, ws=ws)

    win = fspecial(fltr_specs, ws=ws)
    win = np.ones((ws, ws)) / ws**2

    gt_sum_sq, im_sum_sq, gt_im_sum_mul = _get_sums(gt, im, win, mode)
    sigmagt_sq, sigmaim_sq, sigmagt_im = _get_sigmas(
        gt, im, win, mode, sums=(gt_sum_sq, im_sum_sq, gt_im_sum_mul))

    assert C1 > 0
    assert C2 > 0

    ssim_map = ((2 * gt_im_sum_mul + C1) * (2 * sigmagt_im + C2)) / \
        ((gt_sum_sq + im_sum_sq + C1) * (sigmagt_sq + sigmaim_sq + C2))
    cs_map = (2 * sigmagt_im + C2) / (sigmagt_sq + sigmaim_sq + C2)

    return np.mean(ssim_map)


# Pixel Based Visual Information Fidelity
def PBVIF(gt, im, sigma_nsq=2):
    EPS = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2.0**(4 - scale + 1) + 1
        win = fspecial(Filter.GAUSSIAN, ws=N, sigma=N / 5)

        if scale > 1:
            gt = filter2(gt, win, 'valid')[::2, ::2]
            im = filter2(im, win, 'valid')[::2, ::2]

        gt_sum_sq, im_sum_sq, gt_im_sum_mul = _get_sums(
            gt, im, win, mode='valid')
        sigmagt_sq, sigmaim_sq, sigmagt_im = _get_sigmas(
            gt, im, win, mode='valid', sums=(gt_sum_sq,
                                             im_sum_sq, gt_im_sum_mul))

        sigmagt_sq[sigmagt_sq < 0] = 0
        sigmaim_sq[sigmaim_sq < 0] = 0

        g = sigmagt_im / (sigmagt_sq + EPS)
        sv_sq = sigmaim_sq - g * sigmagt_im

        g[sigmagt_sq < EPS] = 0
        sv_sq[sigmagt_sq < EPS] = sigmaim_sq[sigmagt_sq < EPS]
        sigmagt_sq[sigmagt_sq < EPS] = 0

        g[sigmaim_sq < EPS] = 0
        sv_sq[sigmaim_sq < EPS] = 0

        sv_sq[g < 0] = sigmaim_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= EPS] = EPS

        num += np.sum(np.log10(1.0 + (g**2.) *
                               sigmagt_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigmagt_sq / sigma_nsq))

    return num / den


# IOU Intersection over Function
def IOU(gt, im):
    intersection = np.logical_and(gt, im)
    union = np.logical_or(gt, im)
    iou_score = np.sum(intersection) / np.sum(union)


# Confusion confusion_matrix for 2D images
def _conf_mat(gt, im):
    return skm.confusion_matrix(gt.ravel(), im.ravel())


# Accuracy score
def accuracyScore(gt, im):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    return skm.accuracy_score(gt.ravel(), im.ravel())


# Balanced Accuracy Score
def balancedAccScore(gt, im):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    return skm.balanced_accuracy_score(gt.ravel(), im.ravel())


# F1 score
def F1(gt, im, average='weighted'):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    return skm.f1_score(gt.ravel(), im.ravel(), average=average)


# Hamming Loss
def hamming(gt, im):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    return skm.hamming_loss(gt.ravel(), im.ravel())


# Jaccard index
def jaccard(gt, im, average='weighted'):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    return skm.jaccard_score(gt.ravel(), im.ravel(), average=average)


# Precision score
def precision(gt, im, average='weighted'):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    return skm.precision_score(gt.ravel(), im.ravel(), average=average)


# Recall Score
def recall(gt, im, average='weighted'):
    gt = np.array(gt, dtype=np.uint8)
    im = np.array(im, dtype=np.uint8)
    return skm.recall_score(gt.ravel(), im.ravel(), average=average)


# evaluate all metrics
def evaluate(gt, obj_array):
    n_metrics = 12
    n_img_obj = 12

    metric_names = ['PSNR', 'MI', 'MSE', 'RMSE', 'UQI', 'SSIM', 'PBVIF',
                    'F1', 'Hamming', 'Jaccard',
                    'Prec', 'Recall']

    img_names = ['I0', 'I45', 'I90', 'I135', 'S0', 'S1',
                 'S2', 'AoP', 'DoP', 'Intensity', 'RGB_AoP', 'HSL']

    assoc_names = []

    list_df = []

    gt_images = [np.array(gt.images[0, :, :], dtype=np.float),
                 np.array(gt.images[1, :, :], dtype=np.float),
                 np.array(gt.images[2, :, :], dtype=np.float),
                 np.array(gt.images[3, :, :], dtype=np.float),
                 gt.stokes[0, :, :],
                 gt.stokes[1, :, :],
                 gt.stokes[2, :, :],
                 gt.aop, gt.dop, gt.inte,
                 gt.rgb_aop(), gt.rgb_pola()]

    functions = [PSNR, mutual_information, MSE, RMSE, UQI, SSIM, PBVIF,
                 F1, hamming, jaccard,
                 precision, recall]

    for obj_idx, obj in enumerate(obj_array):
        store = np.zeros((n_metrics, n_img_obj))
        assoc_names.append(obj.method)
        images = [np.array(obj.images[0, :, :], dtype=np.float),
                  np.array(obj.images[1, :, :], dtype=np.float),
                  np.array(obj.images[2, :, :], dtype=np.float),
                  np.array(obj.images[3, :, :], dtype=np.float),
                  obj.stokes[0, :, :],
                  obj.stokes[1, :, :],
                  obj.stokes[2, :, :],
                  obj.aop, obj.dop, obj.inte,
                  obj.rgb_aop(), obj.rgb_pola()]

        for index in range(len(images)):
            for idx_f in range(len(functions)):
                if gt_images[index].ndim == 3:
                    store[idx_f, index] = np.mean([
                        functions[idx_f](gt_images[index][:, :, i],
                                         images[index][:, :, i]) for i in range(gt_images[index].shape[2])])
                else:
                    store[idx_f, index] = functions[idx_f](gt_images[index],
                                                           images[index])
        df = pd.DataFrame(store, index=metric_names, columns=img_names)
        list_df.append(df)
    return (list_df, assoc_names)
