from utils import *
from lib import *
import cv2
#from lib import display


if __name__ == "__main__":

    timer.tic()

    Ratliff = ratliff.Ratliff("images/image_00001.png")

    images_r = Ratliff.process()
    # print(images_r)
    timer.toc()

    display.subplot(images_r, 'i0, i45, i90, i135')

    (inten, aop, dop) = polaparam.polarization(images_r)

    rgb = polaparam.rgb(inten, aop, dop)

    print(inten.shape)
    print(aop.shape)
    print(dop.shape)
    print(rgb.shape)

    repres = polaparam.representation(inten, aop, dop, rgb)

    display.plot(repres, 'Intensity, AoP, DoLP, HSL')
