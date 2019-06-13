import numpy as np
import cv2


def polarization(images):
    """ Property that computes the polar params from the 4 images """
    Js = images
    inten = (Js[0]+Js[1]+Js[2]+Js[3])/2.
    aop = (0.5*np.arctan2(Js[1]-Js[3], Js[0]-Js[2]))
    dop = np.sqrt((Js[1]-Js[3])**2+(Js[0]-Js[2])**2) / \
        (Js[0]+Js[1]+Js[2]+Js[3]+np.finfo(float).eps)*2
    return (inten, aop, dop)


def rgb(inten, aop, dop):
    """ Property that return the RGB representation of the pola params """

    hsv = np.uint8(np.dstack(((aop+np.pi/2)/np.pi*180,
                              dop*255,
                              inten/inten.max()*255)))

    print(hsv.shape)
    #np.savetxt('hsv.csv', hsv, delimiter=",")
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def representation(inten, aop, dop, rgb):
    nbr, nbc = rgb.shape[0], rgb.shape[1]

    fina = np.zeros((nbr*2, nbc*2, 3), dtype='uint8')

    aop_colorHSV = np.uint8(np.dstack(((aop+np.pi/2)/np.pi*180,
                                       np.ones(aop.shape)*255,
                                       np.ones(aop.shape)*255)))
    aop_colorRGB = cv2.cvtColor(aop_colorHSV, cv2.COLOR_HSV2RGB)

    for c in range(3):
        fina[:nbr, :nbc, c] = np.uint8(inten/inten.max()*255)
        fina[:nbr, nbc:, c] = aop_colorRGB[:, :, c]
        fina[nbr:, :nbc, c] = np.uint8(dop*255)
        fina[nbr:, nbc:, c] = rgb[:, :, c]
    return fina
