import os
import time
import cv2
import numpy as np
import pytesseract
from datetime import datetime, timedelta
from string import ascii_letters, digits
import datefinder

class OCR_TimeStamp:
    def __init__(self):
        self.timeOCR = None
        self.prevTimer = -1

        self.need_pyt = True
        self.is_started = False
        self.is_detected = False
        self.framecount = 0
        self.check_every_frames = 200

    def run_pytesseract(self, cropped_img):
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 220,255,0)

        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        mask = np.zeros(gray.shape, dtype="uint8")

        # Filtering out noise and consecutively adding connected characters to the empty mask
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # (cX, cY) = centroids[i]

            keepWidth = w > 5 and w < 19
            keepHeight = h > 10 and h < 27
            keepArea = area > 30 and area < 200
            keepY = y > 2  and y < 41

            if all((keepWidth, keepHeight, keepArea, keepY)):
                componentMask = (labels == i).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)

        # Converting the image to text
        text_ocr = pytesseract.image_to_string(
            mask, lang='eng', 
            config='--psm 6 --oem 1 -c tessedit_char_whitelist=1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )[:-2]

        if not any(c for c in text_ocr if not c.isalnum() and not c.isspace()):
            text_ocr = text_ocr.split(' ')
            if len(text_ocr) == 8:
                # Reorganzing Datetime-stamp for PyTesseract to ignore millisec and adding ':' like 'hrs:min:sec' (This : is present in video but are filtered out as noise)
                time_hr_min_sec = ':'.join(text_ocr[3:6])
                millisec = text_ocr[7]
                text_ocr = text_ocr[0] + " " + text_ocr[1] + " " + text_ocr[2] + " " + time_hr_min_sec + " " + text_ocr[6]
                try:
                    millisec = int(millisec)
                    if (millisec / 100 < 1):
                        ocr_string = f'[OCR] Output from PyTesseract: {text_ocr} {millisec}, '
                        try:
                            matches = datefinder.find_dates(text_ocr)
                            num_matches = 0
                            matched_timeDate = None

                            for f in matches:
                                matched_timeDate = f

                            if matched_timeDate:
                                print(ocr_string + f"Datetime extraction: {matched_timeDate}")
                                if millisec / 10 != 0: # Managing double digit and single digit milliseconds
                                    matched_timeDate = matched_timeDate.replace(microsecond=millisec*10000)
                                else:
                                    matched_timeDate = matched_timeDate.replace(microsecond=millisec*100000)
                                self.need_pyt = False
                                return matched_timeDate
                            else:
                                print(ocr_string + f"No detections")
                            
                        except Exception as e:
                            print(f'[OCR] Exception while reading DateTime: {e}')
                    else:
                        raise ValueError
                except ValueError:
                    print('[OCR] Error reading milliseconds')
        return None

    def run_ocr(self, input):
        if self.framecount == 0: # For the first frame of the video, running PyTesseract
            (c_img, videotimer) = input
            time_from_pytes = self.run_pytesseract(c_img)
            if not self.need_pyt:
                self.timeOCR = time_from_pytes
                self.framecount += 1
                self.prevTimer = videotimer
                print(f'[OCR] Initial time: {self.timeOCR}')#, VideoTimer: {videotimer}, Previous Timer: {self.prevTimer}')

        elif self.need_pyt: # Running this every self.check_every_frames and checking whether sync is retained
            (c_img, videotimer) = input
            time_from_pytes = self.run_pytesseract(c_img)
            if isinstance(time_from_pytes, datetime):
                time_from_pytes = time_from_pytes.replace(microsecond=0)
                current_internal_time = self.timeOCR + timedelta(microseconds=(videotimer-self.prevTimer)*1000)
                current_internal_time_check = current_internal_time.replace(microsecond=0)

                if time_from_pytes == current_internal_time_check:
                    print(f'[OCR] Sync check| Internal: {current_internal_time}')
                    print("[OCR] Sync Successfull !!\n")
                    self.need_pyt = False
                    self.prevTimer = videotimer
                    self.timeOCR = current_internal_time
                    self.framecount += 1
                else:
                    print(f'[OCR] Sync check| Internal: {current_internal_time}')
                    print("[OCR] Sync Error !!! Resynchronizing\n")
                    self.prevTimer = videotimer
                    self.framecount = 0
                    self.need_pyt = True
            else:
                print("[OCR] Sync check failed !!! Invalid Data from PyTesseract. Checking the next frame ...")
                self.timeOCR += timedelta(microseconds=(videotimer-self.prevTimer)*1000)
                self.prevTimer = videotimer
                self.need_pyt = True

        else: # Once initial OCR is successfull, consecutively add time (doesn't call PyTesseract)
            videotimer = input
            self.timeOCR += timedelta(microseconds=(videotimer-self.prevTimer)*1000)
            self.prevTimer = videotimer
            self.framecount += 1
            #print(f"[OCR] Current Time: {self.timeOCR}")
            if self.framecount % (self.check_every_frames - 1) == 0:
                self.need_pyt = True

        return self.timeOCR
