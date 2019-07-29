import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.util import img_as_ubyte
from skimage.draw import circle

SIZE_CHAMBER = 160
MIN_SPACING = 350

# simple difference of gaussian
def dog(im, g1=5.0, g2=3.0, abs=False):
    """difference of gaussian
    
    Parameters
    ----------
    im : input image
        numpy array
    g1 : float, 
        first gaussian sigma, by default 5
    g2 : float, 
        second gaussaian sigma, default 3
    abs : bool, optional
        return absolute difference if True, by default False
    """
    dog = gaussian(im, g1)-gaussian(im, g2)
    if abs:
        return np.abs(dog)
    return dog


def detect_circles(im, canny1=30, canny2=50):
    # compute gradient magnitude
    gradient = dog(im, abs=True)
    gradient = gradient.astype(np.uint8)
    # find cirecles via hough transform
    circles = cv2.HoughCircles(gradient,cv2.HOUGH_GRADIENT,1, MIN_SPACING,
         param1=canny1, param2=canny2, minRadius=int(SIZE_CHAMBER*.85), 
         maxRadius=int(SIZE_CHAMBER*1.15))
    print(f"found circles {circles}")
    # Create label image
    circles = np.uint16(np.around(circles))
    label_im = np.zeros(shape=im.shape, dtype=np.uint16)
    for label, i in enumerate(circles[0,:], start=1):
        rr , cc = circle(i[1],i[0],i[2],im.shape)
        label_im[rr,cc] = label
    return label_im