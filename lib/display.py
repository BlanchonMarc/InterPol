import numpy as np
import cv2


def subplot(images, title):
    row1 = cv2.hconcat([np.array(images[0], dtype='uint8'),
                        np.array(images[1], dtype='uint8')])
    row2 = cv2.hconcat([np.array(images[2], dtype='uint8'),
                        np.array(images[3], dtype='uint8')])

    subplot = cv2.vconcat([row1, row2])

    cv2.imshow(title, subplot)
    cv2.waitKey(0)


def plot(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
