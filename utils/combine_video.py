import cv2

vidcap_1 = cv2.VideoCapture('/home/mobilitylabextreme002/Desktop/outputs/raghav_test_normalSort/raghav_test_normalSort.mp4')
vidcap_2 = cv2.VideoCapture('/home/mobilitylabextreme002/Desktop/outputs/raghav_test_deepSort/raghav_test_deepSort.mp4')

vid_writer = cv2.VideoWriter('/home/mobilitylabextreme002/Desktop/outputs/compare_sort/postprocess_compare.mkv', cv2.VideoWriter_fourcc(*'mp4v'), 30, (2560, 720))
framecount = 0
while (vidcap_1.isOpened() and vidcap_2.isOpened()):
    ret1, frame1 = vidcap_1.read()
    ret2, frame2 = vidcap_2.read()

    if ret1 and ret2:
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (1650, 30)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        
        cv2.rectangle(frame1, (org[0] - 5, 0), (1920, org[1] + 10), (0, 0, 0), -1)
        cv2.putText(frame1, 'SORT', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.rectangle(frame2, (org[0] - 5, 0), (1920, org[1] + 10), (0, 0, 0), -1)
        cv2.putText(frame2, 'DeepSORT', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        framecount += 1
        img_h1 = cv2.hconcat([frame1, frame2])
        img_final = cv2.resize(img_h1, (2560, 720))
        vid_writer.write(img_final)
        print(f'[INFO] Writing frame: {framecount}')

    else:
        break

print('Finished combining video')
vid_writer.release()

# vidcap_1 = cv2.VideoCapture('/home/mobilitylabextreme002/Videos/outputs/video_capture_loop/11_03_2022-07_10_05_conf20_iou45/11_03_2022-07_10_05_conf20_iou45.mkv')
# vidcap_2 = cv2.VideoCapture('/home/mobilitylabextreme002/Videos/outputs/video_capture_loop/11_03_2022-07_10_05_conf40_iou45/11_03_2022-07_10_05_conf40_iou45.mkv')
# vidcap_3 = cv2.VideoCapture('/home/mobilitylabextreme002/Videos/outputs/video_capture_loop/11_03_2022-07_10_05_conf60_iou45/11_03_2022-07_10_05_conf60_iou45.mkv')
# vidcap_4 = cv2.VideoCapture('/home/mobilitylabextreme002/Videos/outputs/video_capture_loop/11_03_2022-07_10_05_conf80_iou45/11_03_2022-07_10_05_conf80_iou45.mkv')

# vid_writer = cv2.VideoWriter('/home/mobilitylabextreme002/Videos/outputs/video_capture_loop/11_03_2022-07_10_05_iou_compare.mkv', cv2.VideoWriter_fourcc(*'mp4v'), 30, (3840, 2160))
# framecount = 0
# while (vidcap_1.isOpened() and vidcap_2.isOpened()):
#     ret1, frame1 = vidcap_1.read()
#     ret2, frame2 = vidcap_2.read()
#     ret3, frame3 = vidcap_3.read()
#     ret4, frame4 = vidcap_4.read()

#     if ret1 and ret2 and ret3 and ret4:
#         # font
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         org = (1650, 30)
#         fontScale = 1
#         color = (0, 0, 255)
#         thickness = 2
        
#         cv2.rectangle(frame1, (org[0] - 5, 0), (1920, org[1] + 10), (0, 0, 0), -1)
#         cv2.putText(frame1, 'Conf_Thres 20%', org, font, 
#                         fontScale, color, thickness, cv2.LINE_AA)
        
#         cv2.rectangle(frame2, (org[0] - 5, 0), (1920, org[1] + 10), (0, 0, 0), -1)
#         cv2.putText(frame2, 'Conf_Thres 40%', org, font, 
#                         fontScale, color, thickness, cv2.LINE_AA)

#         cv2.rectangle(frame3, (org[0] - 5, 0), (1920, org[1] + 10), (0, 0, 0), -1)
#         cv2.putText(frame3, 'Conf_Thres 60%', org, font, 
#                         fontScale, color, thickness, cv2.LINE_AA)

#         cv2.rectangle(frame4, (org[0] - 5, 0), (1920, org[1] + 10), (0, 0, 0), -1)
#         cv2.putText(frame4, 'Conf_Thres 80%', org, font, 
#                         fontScale, color, thickness, cv2.LINE_AA)

#         framecount += 1
#         img_h1 = cv2.hconcat([frame1, frame2])
#         img_h2 = cv2.hconcat([frame3, frame4])
#         img_final = cv2.vconcat([img_h1, img_h2])
#         vid_writer.write(img_final)
#         print(f'[INFO] Writing frame: {framecount}')

#     else:
#         break

# print('Finished combining video')
# vid_writer.release()