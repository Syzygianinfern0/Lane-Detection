import cv2
import numpy as np
import matplotlib.pyplot as plt

# PIC = 'assets/whiteCarLaneSwitch.jpg'  # Select your image here
# SHAPE = cv2.imread(PIC).shape
#
#
# def nothing(x):
#     """
#     Redundant but necessary callback Function
#     :param x: Redundant param
#     :return: None
#     """
#     pass
#
#
# # ~~~~~~~~~~~~~~~~~ AWESOME TRACKBARS ~~~~~~~~~~~~~~~~~ #
# cv2.namedWindow('bars', cv2.WINDOW_NORMAL)  # Window for all TrackBars
# cv2.createTrackbar('Blur', 'bars', 3, 15, nothing)  # Gaussian Blur Kernel
# cv2.createTrackbar('Canny_Low', 'bars', 130, 255, nothing)  # Low Threshold for Canny
# cv2.createTrackbar('Canny_High', 'bars', 250, 255, nothing)  # High Threshold for Canny
# cv2.createTrackbar('ROI_V1_x', 'bars', 420, SHAPE[1], nothing)  # Top Left
# cv2.createTrackbar('ROI_V1_y', 'bars', 290, SHAPE[0], nothing)
# cv2.createTrackbar('ROI_V2_x', 'bars', 560, SHAPE[1], nothing)  # Top Right
# cv2.createTrackbar('ROI_V2_y', 'bars', 290, SHAPE[0], nothing)
# cv2.createTrackbar('ROI_V3_x', 'bars', 910, SHAPE[1], nothing)  # Bottom Right
# cv2.createTrackbar('ROI_V3_y', 'bars', 540, SHAPE[0], nothing)
# cv2.createTrackbar('ROI_V4_x', 'bars', 65, SHAPE[1], nothing)  # Bottom Left
# cv2.createTrackbar('ROI_V4_y', 'bars', 540, SHAPE[0], nothing)
# cv2.createTrackbar('Rho', 'bars', 1, 15, nothing)  # Polar Distance in Hough Coordinates
# cv2.createTrackbar('Theta', 'bars', 1, 15, nothing)  # Polar Angle in Hough Coordinates
# cv2.createTrackbar('Threshold', 'bars', 10, 30, nothing)  # Number of Votes for Selection
# cv2.createTrackbar('Min_Line_Length', 'bars', 20, 100, nothing)  # Min length in Pixels
# cv2.createTrackbar('Max_Line_Gap', 'bars', 1, 50, nothing)  # Max distance between segments
#
#
# # ~~~~~~~~~~~~~~~~~ REGION OF INTEREST ~~~~~~~~~~~~~~~~~ #
# # noinspection PyShadowingNames
# def roi(image, vertices):
#     """
#     Filters the Region of Interest
#     :param image: Base image to filter from
#     :param vertices: A list of ordered numpy arrays
#     :return: The masked image
#     """
#     mask = np.zeros_like(image)  # Dummy to mask on
#     cv2.fillPoly(mask, [np.int32(vertices)], 255)  # Positive Indices
#
#     masked_image = cv2.bitwise_and(image, mask)
#     return masked_image
#
#
# # ~~~~~~~~~~~~~~~~~ LINE APPROXIMATORS ~~~~~~~~~~~~~~~~~ #
# def draw_line(img, x, y, color=(0, 0, 255), thickness=20):
#     """
#     Generates an average polynomial of degree 1 (i.e. a line) from a set of points
#     :param img: Base image to plot on
#     :param x: Array of X coordinates
#     :param y: Array of Y coordinates
#     :param color: RGB of line
#     :param thickness: Pixels thickness
#     :return: None
#     """
#     if len(x) == 0:  # Empty case causes troubles
#         return
#     params = np.polyfit(x, y, 1)  # Generate the m, b for the approximated line
#     m = params[0]
#     b = params[1]
#     maxY = img.shape[0]
#     maxX = img.shape[1]
#     y1 = maxY
#     x1 = int((y1 - b) / m)  # Get the coordinates
#     y2 = int((maxY / 2)) + 60
#     x2 = int((y2 - b) / m)  # Get the coordinates
#     cv2.line(img, (x1, y1), (x2, y2), color, thickness)  # Draw on the base image
#
#
# # noinspection PyShadowingNames
# def draw_lines(img, lines, color=(0, 0, 255), thickness=20):
#     """
#     Approximates Hough Lines to two lines of opposite slopes
#     :param img: Base image to plot on
#     :param lines: Generated Hough Lines
#     :param color: RGB of Lines
#     :param thickness: Pixel Thickness
#     :return: None
#     """
#     # Placeholders for the points
#     leftPointsX = []
#     leftPointsY = []
#     rightPointsX = []
#     rightPointsY = []
#
#     for line in lines:
#         for x1, y1, x2, y2 in line:  # iterate over data points
#             m = (y1 - y2) / (x1 - x2)  # Get the slope
#             if m < 0:  # Classify the Points
#                 leftPointsX.append(x1)
#                 leftPointsY.append(y1)
#                 leftPointsX.append(x2)
#                 leftPointsY.append(y2)
#             else:
#                 rightPointsX.append(x1)
#                 rightPointsY.append(y1)
#                 rightPointsX.append(x2)
#                 rightPointsY.append(y2)
#
#     draw_line(img, leftPointsX, leftPointsY, color, thickness)  # Plot Negative Sloped Line
#
#     draw_line(img, rightPointsX, rightPointsY, color, thickness)  # Plot Positive Sloped Line
#
#
# # ~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~ #
# if __name__ == '__main__':
#     original = cv2.imread(PIC)
#     test_img = cv2.imread(PIC, cv2.IMREAD_GRAYSCALE)
#
#     while True:
#         k = cv2.waitKey(1) & 0xFF  # Escape Key
#         if k == 27:  # Condition of exit
#             break
#
#         # Reading the Trackbars
#         kernel = int(cv2.getTrackbarPos('Blur', 'bars')) * 2 - 1  # Force odd Number for kernel
#         low = cv2.getTrackbarPos('Canny_Low', 'bars')
#         high = cv2.getTrackbarPos('Canny_High', 'bars')
#         vertices = ((cv2.getTrackbarPos('ROI_V1_x', 'bars'),  # Dirty Vertices
#                      cv2.getTrackbarPos('ROI_V1_y', 'bars')),
#                     (cv2.getTrackbarPos('ROI_V2_x', 'bars'),
#                      cv2.getTrackbarPos('ROI_V2_y', 'bars')),
#                     (cv2.getTrackbarPos('ROI_V3_x', 'bars'),
#                      cv2.getTrackbarPos('ROI_V3_y', 'bars')),
#                     (cv2.getTrackbarPos('ROI_V4_x', 'bars'),
#                      cv2.getTrackbarPos('ROI_V4_y', 'bars')))
#         rho = cv2.getTrackbarPos('Rho', 'bars')
#         theta = cv2.getTrackbarPos('Theta', 'bars') * np.pi / 180  # Conversion to radians
#         threshold = cv2.getTrackbarPos('Threshold', 'bars')
#         min_line_len = cv2.getTrackbarPos('Min_Line_Length', 'bars')
#         max_line_gap = cv2.getTrackbarPos('Max_Line_Gap', 'bars')
#
#         # Processing
#         blur_image = cv2.GaussianBlur(test_img, (kernel, kernel), 0)  # Blur to smoothen image
#         canny_image = cv2.Canny(blur_image, low, high)  # Edge Detection
#         roi_img = roi(blur_image, vertices)  # Region masker debug
#         roi_canny = roi(canny_image, vertices)  # Region masker actual
#         lines = cv2.HoughLinesP(roi_canny,  # Generate lines based on intersections in Hough Plane
#                                 rho, theta, threshold,
#                                 np.array([]),
#                                 min_line_len, max_line_gap)
#         line_img = np.zeros((test_img.shape[0], test_img.shape[1], 3), dtype=np.uint8)  # Dummy frame to debug lines
#         draw_lines(line_img, lines)  # Call to function
#
#         result = cv2.addWeighted(original, 0.8, line_img, 1.0, 0)  # Overlay on input image
#
#         # View Results
#         cv2.imshow('test', test_img)
#         cv2.imshow('blur_image', blur_image)
#         cv2.imshow('canny_image', canny_image)
#         cv2.imshow('roi_img', roi_img)
#         cv2.imshow('roi_canny', roi_canny)
#         cv2.imshow('line_img', line_img)
#         cv2.imshow('result', result)
#
#     # Write to disk function
#     cv2.imwrite('results/' + PIC.split('/')[-1], result)
#     cv2.destroyAllWindows()


vidcap = cv2.VideoCapture('assets/solidYellowLeft.mp4')
success, image = vidcap.read()
success = True
count = 0
cv2.namedWindow('test')
while success:
    cv2.imshow('test', image)
    cv2.waitKey(5)
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
cv2.destroyAllWindows()
print(count)
