import numpy as np


# Ordering in consequence of the superpixel order
def ordering(matrix, type_raw, pix_order):

    images = np.zeros((4, matrix.shape[0], matrix.shape[1]))

    for i in range(4):
        images[i, :, :] = np.array(matrix[:, :, i], dtype=np.double)

    return np.asarray(images[pix_order, :, :], dtype=type_raw)
