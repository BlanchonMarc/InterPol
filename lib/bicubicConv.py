import math
import numpy as np

# Function for convolution bicubic interpolation .


def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s)**3) - (a + 3) * (abs(s)**2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s)**3) - (5 * a) * (abs(s)**2) + (8 * a) * abs(s) - 4 * a
    return 0

# padding function for convolution bicubic interpolation


def padding(img, H, W):
    zimg = np.zeros((H + 4, W + 4))
    zimg[2:H + 2, 2:W + 2] = img
    # Pad the first/last two col and row
    zimg[2:H + 2, 0:2] = img[:, 0:1]
    zimg[H + 2:H + 4, 2:W + 2] = img[H - 1:H, :]
    zimg[2:H + 2, W + 2:W + 4] = img[:, W - 1:W]
    zimg[0:2, 2:W + 2] = img[0:1, :]
    # Pad the missing eight points
    zimg[0:2, 0:2] = img[0, 0]
    zimg[H + 2:H + 4, 0:2] = img[H - 1, 0]
    zimg[H + 2:H + 4, W + 2:W + 4] = img[H - 1, W - 1]
    zimg[0:2, W + 2:W + 4] = img[0, W - 1]
    return zimg

# Bicubic interpolation


def bicubic(img, ratio, a):
    # Get image size
    H, W = img.shape

    img = padding(img, H, W)
    # Create new image
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)
    dst = np.zeros((dH, dW))

    h = 1 / ratio

    inc = 0

    for j in range(dH):
        for i in range(dW):
            x, y = i * h + 2, j * h + 2

            x1 = 1 + x - math.floor(x)
            x2 = x - math.floor(x)
            x3 = math.floor(x) + 1 - x
            x4 = math.floor(x) + 2 - x

            y1 = 1 + y - math.floor(y)
            y2 = y - math.floor(y)
            y3 = math.floor(y) + 1 - y
            y4 = math.floor(y) + 2 - y

            mat_l = np.matrix(
                [[u(x1, a**1), u(x2, a**2), u(x3, a**3), u(x4, a**4)]])
            mat_m = np.matrix([[img[int(y - y1), int(x - x1)], img[int(y - y2), int(x - x1)], img[int(y + y3), int(x - x1)], img[int(y + y4), int(x - x1)]],
                               [img[int(y - y1), int(x - x2)], img[int(y - y2), int(x - x2)],
                                img[int(y + y3), int(x - x2)], img[int(y + y4), int(x - x2)]],
                               [img[int(y - y1), int(x + x3)], img[int(y - y2), int(x + x3)],
                                img[int(y + y3), int(x + x3)], img[int(y + y4), int(x + x3)]],
                               [img[int(y - y1), int(x + x4)], img[int(y - y2), int(x + x4)], img[int(y + y3), int(x + x4)], img[int(y + y4), int(x + x4)]]])

            mat_r = np.matrix([[u(y1, a**1)], [u(y2, a**2)],
                               [u(y3, a**3)], [u(y4, a**4)]])
            dst[j, i] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst


def conv_bicubic(raw):
    I = np.array(raw, dtype=np.double)

    I0 = I[0::2, 0::2]
    I45 = I[0::2, 1::2]
    I90 = I[1::2, 1::2]
    I135 = I[1::2, 0::2]
    I0 = bicubic(I0, 2, 1 / 2)
    I45 = bicubic(I45, 2, 1 / 2)
    I90 = bicubic(I90, 2, 1 / 2)
    I135 = bicubic(I135, 2, 1 / 2)

    return np.dstack((I0, I45, I90, I135))
