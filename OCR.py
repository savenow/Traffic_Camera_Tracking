"""
Main purpose of this file is to extract the Date-Timestamp information from the video
"""

import pytesseract
import cv2, PIL
import numpy as np
import os

def dispImg(THRESH):
    """
    Display the thresholded image

    Args:
        a cv2 image file (mostly the thresholded image)
    """
    cv2.imshow("THRESH", THRESH)
    cv2.waitKey(0)

def OCR_image(img_path, display_image=False):
    """
    Extracts date and time information for a single image. This function is meant to be imported into the main inference file
    for each frame of the video

    Args:
        img_path (str): Path of the image
        display_image (bool, optional): Decides whether to display the image for reference or not. Defaults to False.

    Returns:
        date_time (str): a string containing the date and time
    """
    
    img = cv2.imread(img_path)
    
    # This is the specific upper left hand side coordinates of the rectangle containing the date and time. If this position changes in the video,
    # the given coordinates must be changed appropriately
    # The coordinates of the rectangle used here: (0, 0) and (570, 40)
    text_roi = img[0:40, 0:570]

    # convert to gray
    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image - The values are found out by trial and error method
    ret, thresh = cv2.threshold(gray,237,255,0)

    # Extracts the text using pytesseract library
    date_time = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')[4:-1]
    
    if display_image:
        dispImg(thresh)
    
    return date_time

def LoopThroughFiles(dir_path):
    """
    Local testing function to loop over certain image files in a directory

    Args:
        dir_path (str): Path of the directory
    """
    for files in os.listdir(dir_path):
        print(files)
        print(OCR_image(dir_path + '\\' + files))


def main():
    """
    Function used for testing purposes. (Not in use anymore)
    """
    
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

