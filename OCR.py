"""
Main purpose of this file is to extract the Date-Timestamp information from the video
"""

import pytesseract
import cv2, PIL
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\balaji\AppData\Local\Programs\Tesseract-OCR\tesseract'

image_path = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Date_Time_Test\frame_000003.png'
img = cv2.imread(image_path)

# date_time_coordinates: (0, 0) till (570, 40)
text_roi = img[0:40, 0:570]

# gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Morph open to remove noise and invert image
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# invert = 255 - opening

# data = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
# print(data)

# cv2.imshow("Text_ROI", text_roi)
# cv2.imshow('thresh', thresh)
# cv2.imshow('opening', opening)
# cv2.imshow('invert', invert)

hh, ww, cc = text_roi.shape

# convert to gray
gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)

# threshold the grayscale image
ret, thresh = cv2.threshold(gray,165,255,0)

# create black image to hold results
results = np.zeros((hh,ww))

# find contours
cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

# Contour filtering and copy contour interior to new black image.
for c in cntrs:
    area = cv2.contourArea(c)
    if area > 1000:
        x,y,w,h = cv2.boundingRect(c)
        results[y:y+h,x:x+w] = thresh[y:y+h,x:x+w]

# invert the results image so that have black letters on white background
results = (255 - results)

# write results to disk
#cv2.imwrite("numbers_extracted.png", results)

data = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
print(data)

cv2.imshow("THRESH", thresh)
cv2.imshow("RESULTS", results)
cv2.waitKey(0)
