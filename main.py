import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import cv2

# Open the first camera device (usually device 0)
camera = cv2.VideoCapture(0)

if camera.isOpened():
    # Read a frame from the camera
    ret, frame = camera.read()

    if ret:
        # Save the captured frame as an image file
        cv2.imwrite('captured_image.jpg', frame)

    # Release the camera
    camera.release()
else:
    print("Unable to open camera")



# read input image
img = cv2.imread('captured_image.jpg')

# convert the input image to a grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

# if chessboard corners are detected
if ret == True:
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (7, 7), corners, ret)
    cv2.imshow('Chessboard', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
# cap = cv.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# ret, frame = cap.read()
#
# if not ret:
#     print("Error: Could not read frame.")
#     exit()
#
# cv.imshow('Captured Image', frame)
#
# cv.imwrite('captured_image.jpg', frame)
#
# img = cv.imread('captured_image.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
#
# laplacian = cv.Laplacian(img, cv.CV_64F)
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
#
# plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#
# plt.show()
#
# cv.waitKey(0)
#
# cap.release()
# cv.destroyAllWindows()