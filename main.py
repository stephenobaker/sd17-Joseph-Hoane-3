import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame.")
    exit()

cv2.imshow('Captured Image', frame)

cv2.imwrite('captured_image.jpg', frame)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()