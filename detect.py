import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO = 'assets/dashcamvid.mp4'
PIC = 'assets/dashcampic.png'

if __name__ == '__main__':
    test_img = cv2.imread(PIC, cv2.IMREAD_GRAYSCALE)
    blur_image = cv2.GaussianBlur(test_img, (3, 3), 0)
    canny_image = cv2.Canny(blur_image, 100, 150)
    cv2.imshow('test', test_img)
    cv2.imshow('blur_image', blur_image)
    cv2.imshow('canny_image', canny_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

