import cv2
import numpy as np
import matplotlib.pyplot as plt

PIC = 'assets/solidWhiteCurve.jpg'
SHAPE = cv2.imread(PIC).shape


def nothing(x):
    pass


cv2.namedWindow('bars')
cv2.createTrackbar('Blur', 'bars', 1, 15, nothing)
cv2.createTrackbar('Canny_Low', 'bars', 50, 255, nothing)
cv2.createTrackbar('Canny_High', 'bars', 150, 255, nothing)
cv2.createTrackbar('ROI_V1_x', 'bars', 0, SHAPE[0], nothing)  # Top Left
cv2.createTrackbar('ROI_V1_y', 'bars', 0, SHAPE[1], nothing)
cv2.createTrackbar('ROI_V2_x', 'bars', SHAPE[0], SHAPE[0], nothing)  # Top Right
cv2.createTrackbar('ROI_V2_y', 'bars', 0, SHAPE[1], nothing)
cv2.createTrackbar('ROI_V3_x', 'bars', SHAPE[0], SHAPE[0], nothing)  # Bottom Right
cv2.createTrackbar('ROI_V3_y', 'bars', SHAPE[1], SHAPE[1], nothing)
cv2.createTrackbar('ROI_V4_x', 'bars', 0, SHAPE[0], nothing)  # Bottom Left
cv2.createTrackbar('ROI_V4_y', 'bars', SHAPE[1], SHAPE[1], nothing)


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
    blur_image = cv2.GaussianBlur(test_img, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)

    roi_img = roi(canny_image)

    cv2.imshow('test', test_img)
    cv2.imshow('blur_image', blur_image)
    cv2.imshow('canny_image', canny_image)
    cv2.imshow('roi_img', roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
