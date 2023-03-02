# import numpy as np
# import cv2 as cv
# import os

# main_path = "/home/mobilitylabextreme002/Downloads/yt_traffic.mp4" 
# cap = cv.VideoCapture(main_path)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
   
# size = (frame_width, frame_height)
   
# # Below VideoWriter object will create
# # a frame of above defined The output 
# # is stored in 'filename.avi' file.
# # vid_writer = cv.VideoWriter('outputs/optical_flow_raghav_clipped.avi', 
# #                          cv.VideoWriter_fourcc(*'MJPG'),
# #                          30, size)
# ret, frame1 = cap.read()
# prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[..., 1] = 255
# frame_count = 1
# while(1):
#     print(f"Processed frame number: {frame_count}")
#     frame_count += 1
#     ret, frame2 = cap.read()
#     if not ret:
#         print('No frames grabbed!')
#         break
#     next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#     flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#     bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     # vid_writer.write(bgr)
#     cv.imshow('frame2', bgr)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv.imwrite('opticalfb.png', frame2)
#         cv.imwrite('opticalhsv.png', bgr)
#     prvs = next
# # vid_writer.release()
# cv.destroyAllWindows()

# import numpy as np
# import cv2
  
# cap = cv2.VideoCapture('sample.mp4')
  
# # params for corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
  
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                               10, 0.03))
  
# # Create some random colors
# color = np.random.randint(0, 255, (100, 3))
  
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame,
#                         cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
#                              **feature_params)
  
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
  
# while(1):
      
#     ret, frame = cap.read()
#     frame_gray = cv2.cvtColor(frame,
#                               cv2.COLOR_BGR2GRAY)
  
#     # calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
#                                            frame_gray,
#                                            p0, None,
#                                            **lk_params)
  
#     # Select good points
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]
  
#     # draw the tracks
#     for i, (new, old) in enumerate(zip(good_new, 
#                                        good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         mask = cv2.line(mask, (a, b), (c, d),
#                         color[i].tolist(), 2)
          
#         frame = cv2.circle(frame, (a, b), 5,
#                            color[i].tolist(), -1)
          
#     img = cv2.add(frame, mask)
  
#     cv2.imshow('frame', img)
      
#     k = cv2.waitKey(25)
#     if k == 27:
#         break
  
#     # Updating Previous frame and points 
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)
  
# cv2.destroyAllWindows()
# cap.release()


# Lukas-Kanade Optical Flow
import numpy as np
import cv2
  
# cap = cv2.VideoCapture('/home/mobilitylabextreme002/Downloads/yt_traffic.mp4')
cap = cv2.VideoCapture('/home/mobilitylabextreme002/Videos/small_clipped/raghav_calibration/raghav_clipped.mp4')
  
# params for corner detection
feature_params = dict(
    maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7 
)
  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))
  
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
  
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,
                        cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
                             **feature_params)
  
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
  
while(1):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
  
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           p0, None,
                                           **lk_params)
  
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
  
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, 
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d),
                        color[i].tolist(), 2)
          
        frame = cv2.circle(frame, (a, b), 5,
                           color[i].tolist(), -1)
          
    img = cv2.add(frame, mask)
  
    cv2.imshow('frame', img)
      
    k = cv2.waitKey(0)
    if k == 27:
        break
  
    # Updating Previous frame and points 
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
  
cv2.destroyAllWindows()
cap.release()