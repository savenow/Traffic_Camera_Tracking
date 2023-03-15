import cv2
import numpy as np

def motion_detector(cap): 
    frame_count = 0
    previous_frame = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Specify the video codec
    out = cv2.VideoWriter('outputs/raghav_clipped_opflow.mp4', fourcc, 30.0, (1280, 720)) # Specify the output filename, codec, frame rate, and frame size
    out_thresh = cv2.VideoWriter('outputs/raghav_clipped_opflow_thresh.mp4', fourcc, 30.0, (1280, 720)) # Specify the output filename, codec, frame rate, and frame size
   
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            print(f"Processing frame: {frame_count}")
            
            # 1. Load image; convert to RGB
            # img_rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

            if ((frame_count % 1) == 0):
                # 2. Prepare image; grayscale and blur
                prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)
            
                if frame_count == 1:
                    previous_frame = prepared_frame
                    continue
                # calculate difference and update previous frame
                diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
                previous_frame = prepared_frame

                # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
                kernel = np.ones((5, 5))
                diff_frame = cv2.dilate(diff_frame, kernel, 1)

                # 5. Only take different areas that are different enough (>20 / 255)
                thresh_frame = cv2.threshold(src=diff_frame, thresh=10, maxval=255, type=cv2.THRESH_BINARY)[1]

                contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 200:
                        # too small: skip!
                        continue
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                #cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                # cv2.resize(thresh_frame, (1280, 720))
                frame = cv2.resize(frame, (1280, 720))
                out.write(frame)
                out_thresh.write(cv2.resize(cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR), (1280, 720)))
                # cv2.imshow("thresh", thresh_frame)
                # cv2.imshow("countour", img_rgb)
                #cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                # cv2.resize(thresh_frame, (1280, 720))
                # if k == 27:
                #     break

        else:
            break
    
    out.release()

cap = cv2.VideoCapture("/home/mobilitylabextreme002/Videos/small_clipped/raghav_calibration/raghav_clipped.mp4")
motion_detector(cap)
