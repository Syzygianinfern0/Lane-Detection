import cv2
import numpy as np
import matplotlib.pyplot as plt

PIC = 'assets/solidYellowLeft.jpg'
SHAPE = cv2.imread(PIC).shape


def nothing(x):
    pass


cv2.namedWindow('bars', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Blur', 'bars', 3, 15, nothing)
cv2.createTrackbar('Canny_Low', 'bars', 130, 255, nothing)
cv2.createTrackbar('Canny_High', 'bars', 250, 255, nothing)
cv2.createTrackbar('ROI_V1_x', 'bars', 420, SHAPE[1], nothing)  # Top Left
cv2.createTrackbar('ROI_V1_y', 'bars', 290, SHAPE[0], nothing)
cv2.createTrackbar('ROI_V2_x', 'bars', 560, SHAPE[1], nothing)  # Top Right
cv2.createTrackbar('ROI_V2_y', 'bars', 290, SHAPE[0], nothing)
cv2.createTrackbar('ROI_V3_x', 'bars', 910, SHAPE[1], nothing)  # Bottom Right
cv2.createTrackbar('ROI_V3_y', 'bars', 540, SHAPE[0], nothing)
cv2.createTrackbar('ROI_V4_x', 'bars', 65, SHAPE[1], nothing)  # Bottom Left
cv2.createTrackbar('ROI_V4_y', 'bars', 540, SHAPE[0], nothing)


# noinspection PyShadowingNames
def roi(image, vertices):
    # top_right = [810, 430]
    # top_left = [670, 430]
    # bottom_left = [380, 600]
    # bottom_right = [1020, 600]
    # # noinspection PyShadowingNames
    # vertices = [np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.int32)]

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.int32(vertices)], 255)
    # cv2.fillPoly(mask, [np.int32(np.flip(vertices, axis=0))], 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


if __name__ == '__main__':
    test_img = cv2.imread(PIC, cv2.IMREAD_GRAYSCALE)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        kernel = int(cv2.getTrackbarPos('Blur', 'bars')) * 2 - 1
        low = cv2.getTrackbarPos('Canny_Low', 'bars')
        high = cv2.getTrackbarPos('Canny_High', 'bars')
        vertices = ((cv2.getTrackbarPos('ROI_V1_x', 'bars'),
                     cv2.getTrackbarPos('ROI_V1_y', 'bars')),
                    (cv2.getTrackbarPos('ROI_V2_x', 'bars'),
                     cv2.getTrackbarPos('ROI_V2_y', 'bars')),
                    (cv2.getTrackbarPos('ROI_V3_x', 'bars'),
                     cv2.getTrackbarPos('ROI_V3_y', 'bars')),
                    (cv2.getTrackbarPos('ROI_V4_x', 'bars'),
                     cv2.getTrackbarPos('ROI_V4_y', 'bars')))

        blur_image = cv2.GaussianBlur(test_img, (kernel, kernel), 0)
        canny_image = cv2.Canny(blur_image, low, high)
        roi_img = roi(blur_image, vertices)
        roi_canny = roi(canny_image, vertices)

        cv2.imshow('test', test_img)
        cv2.imshow('blur_image', blur_image)
        cv2.imshow('canny_image', canny_image)
        cv2.imshow('roi_img', roi_img)
        cv2.imshow('roi_canny', roi_canny)
    cv2.destroyAllWindows()
