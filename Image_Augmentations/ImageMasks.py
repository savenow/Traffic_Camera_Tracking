import cv2
import numpy as np

blank_image_1 = np.zeros((500, 500), dtype="uint8")
blank_image_2 = np.zeros((500, 500), dtype="uint8")
cv2.circle(blank_image_1, (250, 250), 150, 255, -1)
blank_image_2
cv2.imshow("Circle", blank_image_1)
cv2.imshow('Empty', blank_image_2)
cv2.waitKey(0)
