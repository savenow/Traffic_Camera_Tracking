"""
Main purpose of this script is to split the video files in a directory into images.
The images will stored be in the format: 'frame_000001.png', 'frame_000002.png' ..... 

IMPORTANT: The path of ffmpeg.exe (refer 'ffmpeg_path') must be configured properly for this to work
"""

import os

# Change this main input path to your specific video directory
input_path = os.path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new')
ffmpeg_path = os.path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\ffmpeg.exe')

def createFrames():
    for dir in os.listdir(input_path):
        if os.path.isdir(input_path + '\\' + dir) and dir in ['49', '50', '51']:
            clip_path = input_path + f'\\{dir}'
        
            print(f'Processing File: {clip_path}')

            image_path = clip_path + '\\images'
            if not os.path.isdir(image_path):
                os.makedirs(image_path)

            is_video_present = False
            for files in os.listdir(input_path + '\\' + dir):
                if files[-4:] in ['.mkv', '.mp4', '.avi', '.mov']:
                    clip_path += f'\\{files}'
                    is_video_present = True

                    ffmpeg_command = ffmpeg_path + ' -i ' + clip_path + " " + image_path + '\\frame_%06d.png'
                    print(ffmpeg_command)
                    os.system(ffmpeg_command)

            if not is_video_present:
                print('The video file couldnot be found in the given path.')
        
createFrames()


