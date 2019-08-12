import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO = 'assets/dashcamvid.mp4'
PIC = 'assets/dashcampic.png'


def roi(image):
    top_right = [810, 430]
    top_left = [670, 430]
    bottom_left = [380, 600]
    bottom_right = [1020, 600]
    vertices = [np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.int32)]

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


if __name__ == '__main__':
    test_img = cv2.imread(PIC, cv2.IMREAD_GRAYSCALE)
    blur_image = cv2.GaussianBlur(test_img, (3, 3), 0)
    canny_image = cv2.Canny(blur_image, 100, 150)

    roi_img = roi(canny_image)

    cv2.imshow('test', test_img)
    cv2.imshow('blur_image', blur_image)
    cv2.imshow('canny_image', canny_image)
    cv2.imshow('roi_img', roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
