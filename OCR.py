"""
Main purpose of this file is to extract the Date-Timestamp information from the video
"""

import pytesseract
import cv2, PIL
import numpy as np
import os

def dispImg(THRESH):
    cv2.imshow("THRESH", THRESH)
    cv2.waitKey(0)

def OCR_image(img_path, display_image=False):
    img = cv2.imread(img_path)
    # date_time_coordinates: (0, 0) till (570, 40)
    text_roi = img[0:40, 0:570]

    # convert to gray
    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image - The values are found out by trial and error method
    ret, thresh = cv2.threshold(gray,237,255,0)

    date_time = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')[4:-1]
    
    if display_image:
        dispImg(thresh)
    
    return date_time

def extract_date_time(frame):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\balaji\AppData\Local\Programs\Tesseract-OCR\tesseract'

    # date_time_coordinates: (0, 0) till (570, 40)
    text_roi = frame[0:40, 0:570]

    # convert to gray
    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image - The values are found out by trial and error method
    ret, thresh = cv2.threshold(gray,237,255,0)

    date_time = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')[4:-1]
    return date_time

def LoopThroughFiles(dir_path):
    for files in os.listdir(dir_path):
        print(files)
        print(OCR_image(dir_path + '\\' + files))


def main():
    # Path for remote PC

    # Path to PyTesseract CMD
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\balaji\AppData\Local\Programs\Tesseract-OCR\tesseract'
    #image_path = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Date_Time_Test\frame_000003.png'

    # Path for my personal Windows PC

    # Path to PyTesseract CMD
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    image_path = os.path.abspath(r'.\Sample_Pictures\OCR\Zoomed_in_3.png')
    dir_path = os.path.abspath(r'.\Sample_Pictures\OCR\\')

    LoopThroughFiles(dir_path)

