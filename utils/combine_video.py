import cv2

vidcap_1 = cv2.VideoCapture('/media/mydisk/videos/output_e800/test_day_3.mp4')
vidcap_2 = cv2.VideoCapture('/media/mydisk/videos/output_e800/test_day_3_65.mp4')
vid_writer = cv2.VideoWriter('/media/mydisk/videos/output_e800/test_day_3_combined_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (3840, 1080))
framecount = 0
while (vidcap_1.isOpened() and vidcap_2.isOpened()):
    ret1, frame1 = vidcap_1.read()
    ret2, frame2 = vidcap_2.read()

    if ret1 and ret2:
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (1650, 30)
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2
        
        cv2.rectangle(frame2, (org[0] - 5, 0), (1920, org[1] + 10), (0, 0, 0), -1)
        cv2.putText(frame2, 'Conf_Thres 65%', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        framecount += 1
        img_h = cv2.hconcat([frame1, frame2])
        vid_writer.write(img_h)
        print(f'[INFO] Writing frame: {framecount}')

    else:
        break

print('Finished combining video')
vid_writer.release()