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
            config='--psm 6 --oem 1'#-c tessedit_char_whitelist=1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )[:-2]

        if not any(c for c in text_ocr if not c.isalnum() and not c.isspace()):
            text_ocr = text_ocr.split(' ')
            print(text_ocr)
            # if len(text_ocr) == 8:
            if len(text_ocr) == 7:
                # Reorganzing Datetime-stamp for PyTesseract to ignore millisec and adding ':' like 'hrs:min:sec' (This : is present in video but are filtered out as noise)
                time_hr_min_sec = ':'.join(text_ocr[3:6])
                millisec = text_ocr[6]
                text_ocr = text_ocr[0] + " " + text_ocr[1] + " " + text_ocr[2] + " " + time_hr_min_sec
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
    
    def run_pytesseract_masked(self, cropped_img):
        print("[DEBUG] Running masked OCR")
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
            config='--psm 6 --oem 1'#-c tessedit_char_whitelist=1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )[:-1]


        if not any(c for c in text_ocr if not c.isalnum() and not c.isspace()):
            text_ocr_modified = ''.join(text_ocr.split(' '))    
            
            if len(text_ocr_modified) in [16]: # For days with only one digit (first 9 days of a month)
                day = int(text_ocr_modified[0])
                month = int(text_ocr_modified[1:3])
                year = int(text_ocr_modified[3:7])
                hr = int(text_ocr_modified[7:9])
                min = int(text_ocr_modified[9:11])
                sec = int(text_ocr_modified[11:13])
                microSec = int(text_ocr_modified[13:16]) * 1000

                extracted_date = datetime(year, month, day, hr, min, sec, microSec)
                self.need_pyt = False
                return extracted_date
            
            elif len(text_ocr_modified) in [17]: # For rest of the days
                day = int(text_ocr_modified[0:2])
                month = int(text_ocr_modified[2:4])
                year = int(text_ocr_modified[4:8])
                hr = int(text_ocr_modified[8:10])
                min = int(text_ocr_modified[10:12])
                sec = int(text_ocr_modified[12:14])
                microSec = int(text_ocr_modified[14:17]) * 1000

                extracted_date = datetime(year, month, day, hr, min, sec, microSec)
                self.need_pyt = False
                return extracted_date

        return None

    
    def run_pytesseract_noMilliSec(self, cropped_img):
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
            config='--psm 6 --oem 1'#-c tessedit_char_whitelist=1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )[:-2]

        if not any(c for c in text_ocr if not c.isalnum() and not c.isspace()):
            text_ocr = text_ocr.split(' ')
            if len(text_ocr) == 7:
                # Reorganzing Datetime-stamp for PyTesseract to ignore millisec and adding ':' like 'hrs:min:sec' (This : is present in video but are filtered out as noise)
                time_hr_min_sec = ':'.join(text_ocr[3:6])               
                text_ocr = text_ocr[0] + " " + text_ocr[1] + " " + text_ocr[2] + " " + time_hr_min_sec # + " " + text_ocr[6]
                
                ocr_string = f'[OCR] Output from PyTesseract: {text_ocr}'
                try:
                    matches = datefinder.find_dates(text_ocr)
                    num_matches = 0
                    matched_timeDate = None

                    for f in matches:
                        matched_timeDate = f

                    if matched_timeDate:
                        print(ocr_string + f"Datetime extraction: {matched_timeDate}")
                        self.need_pyt = False
                        return matched_timeDate
                    else:
                        print(ocr_string + f"No detections")
                    
                except Exception as e:
                    print(f'[OCR] Exception while reading DateTime: {e}')
                          
        return None

    def run_ocr(self, input, mode='masked'):
        if self.framecount == 0: # For the first frame of the video, running PyTesseract
            (c_img, videotimer) = input
            if mode == 'withMilliSec':
                time_from_pytes = self.run_pytesseract(c_img)    
            elif mode == 'withoutMilliSec':
                time_from_pytes = self.run_pytesseract_noMilliSec(c_img)
            elif mode == 'masked':
                time_from_pytes = self.run_pytesseract_masked(c_img)
            
            if not self.need_pyt:
                self.timeOCR = time_from_pytes
                self.framecount += 1
                self.prevTimer = videotimer
                print(f'[OCR] Initial time: {self.timeOCR}')#, VideoTimer: {videotimer}, Previous Timer: {self.prevTimer}')

        elif self.need_pyt: # Running this every self.check_every_frames and checking whether sync is retained
            (c_img, videotimer) = input
            if mode == 'withMilliSec':
                time_from_pytes = self.run_pytesseract(c_img)
            elif mode == 'withoutMilliSec':
                time_from_pytes = self.run_pytesseract_noMilliSec(c_img)
            elif mode == 'masked':
                time_from_pytes = self.run_pytesseract_masked(c_img)

            
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
            if self.framecount % (self.check_every_frames - 1) == 0:
                self.need_pyt = True

        return self.timeOCR

if __name__ == "__main__":
    ocr = OCR_TimeStamp()
    # video_path = '/home/mobilitylabextreme002/Videos/fkk_new_videos/20220913_074500/20220913_074500_000.mp4'
    video_path = '/home/mobilitylabextreme002/Videos/small_clipped/raghav_calibration/raghav_clipped.mp4'
    # video_path = '/home/mobilitylabextreme002/Videos/small_clipped/test_for_state_light.mkv'
    debug_output_path = "outputs/debug_ocr"
    vid_cap = cv2.VideoCapture(video_path)
    frame_count = 0
    os.makedirs(debug_output_path, exist_ok=True)

    while (vid_cap.isOpened()):
        # if vid_cap.isOpened():
        ret, frame = vid_cap.read()
        if ret:
            frame_count += 1
            
            frame = frame[4:44, 0:410]
            if frame_count == 1:
                cv2.imwrite(os.path.join(debug_output_path, "debug_ocr_crop.png"), frame)

            output = ocr.debug_run_pytesseract(frame)
            # if output == None:
            #     cv2.imwrite(os.path.join(debug_output_path, f"debug_ocr_error_frame{frame_count}.png"), frame)
            # # print(output)
        
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    
    vid_cap.release()
    cv2.destroyAllWindows()

