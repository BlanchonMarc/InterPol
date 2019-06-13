from utils import *
from lib import *
import cv2
import numpy as np
# from lib import display


if __name__ == "__main__":

    """
    Ratliff
    """

    timer.tic()

    Ratliff = ratliff.Ratliff("images/image_00001.tiff")

    images_r = Ratliff.process()

    timer.toc()

    display.subplot(images_r, 'i0, i45, i90, i135')

    (inten, aop, dop) = polaparam.polarization(images_r)

    rgb = polaparam.rgb(inten, aop, dop)

    repres = polaparam.representation(inten, aop, dop, rgb)

    display.plot(repres, 'Intensity, AoP, DoLP, HSL')

    """
    Bicubic
    """
    timer.tic()
    Bicubic = bicubic.Bicubic("images/image_00001.tiff")

    images_r = Bicubic.process()

    timer.toc()

    display.subplot(images_r, 'i0, i45,i 90, i135')

    (inten, aop, dop) = polaparam.polarization(images_r)

    # test
    #np.savetxt("aop.csv", aop, delimiter=",")
    #np.savetxt("dop.csv", dop, delimiter=",")
    #np.savetxt("inten.csv", inten, delimiter=",")
    # display.plot(aop, 'AoP')

    rgb = polaparam.rgb(inten, aop, dop)

    repres = polaparam.representation(inten, aop, dop, rgb)

    display.plot(repres, 'Intensity, AoP, DoLP, HSL')
