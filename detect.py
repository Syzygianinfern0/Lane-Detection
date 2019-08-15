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
cv2.createTrackbar('Rho', 'bars', 1, 15, nothing)
cv2.createTrackbar('Theta', 'bars', 1, 15, nothing)
cv2.createTrackbar('Threshold', 'bars', 10, 30, nothing)
cv2.createTrackbar('Min_Line_Length', 'bars', 20, 100, nothing)
cv2.createTrackbar('Max_Line_Gap', 'bars', 1, 50, nothing)


# noinspection PyShadowingNames
def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.int32(vertices)], 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def drawLine(img, x, y, color=[255, 0, 0], thickness=20):
    if len(x) == 0:
        return
    lineParameters = np.polyfit(x, y, 1)
    m = lineParameters[0]
    b = lineParameters[1]
    maxY = img.shape[0]
    maxX = img.shape[1]
    y1 = maxY
    x1 = int((y1 - b) / m)
    y2 = int((maxY / 2)) + 60
    x2 = int((y2 - b) / m)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# noinspection PyShadowingNames
def draw_lines(img, lines, color=[0, 0, 255], thickness=20):
    leftPointsX = []
    leftPointsY = []
    rightPointsX = []
    rightPointsY = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y1 - y2) / (x1 - x2)
            if m < 0:
                leftPointsX.append(x1)
                leftPointsY.append(y1)
                leftPointsX.append(x2)
                leftPointsY.append(y2)
            else:
                rightPointsX.append(x1)
                rightPointsY.append(y1)
                rightPointsX.append(x2)
                rightPointsY.append(y2)

    drawLine(img, leftPointsX, leftPointsY, color, thickness)

    drawLine(img, rightPointsX, rightPointsY, color, thickness)


if __name__ == '__main__':
    original = cv2.imread(PIC)
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
        rho = cv2.getTrackbarPos('Rho', 'bars')
        theta = cv2.getTrackbarPos('Theta', 'bars') * np.pi / 180
        threshold = cv2.getTrackbarPos('Threshold', 'bars')
        min_line_len = cv2.getTrackbarPos('Min_Line_Length', 'bars')
        max_line_gap = cv2.getTrackbarPos('Max_Line_Gap', 'bars')

        blur_image = cv2.GaussianBlur(test_img, (kernel, kernel), 0)
        canny_image = cv2.Canny(blur_image, low, high)
        roi_img = roi(blur_image, vertices)
        roi_canny = roi(canny_image, vertices)
        lines = cv2.HoughLinesP(roi_canny,
                                rho, theta, threshold,
                                np.array([]),
                                min_line_len, max_line_gap)
        line_img = np.zeros((test_img.shape[0], test_img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines)

        result = cv2.addWeighted(original, 0.8, line_img, 1.0, 0)

        cv2.imshow('test', test_img)
        cv2.imshow('blur_image', blur_image)
        cv2.imshow('canny_image', canny_image)
        cv2.imshow('roi_img', roi_img)
        cv2.imshow('roi_canny', roi_canny)
        cv2.imshow('line_img', line_img)
        cv2.imshow('result', result)
    cv2.destroyAllWindows()

