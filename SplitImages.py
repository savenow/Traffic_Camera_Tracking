import os

input_path = os.path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new')
ffmpeg_path = os.path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\ffmpeg.exe')

def createFrames():
    # main_path refers to current working directory
    for dir in os.listdir(input_path):
        clip_path = input_path + f'\\{dir}'
        #print(clip_path)
        print(f'Processing File: {clip_path}')

        image_path = clip_path + '\\images'
        if not os.path.isdir(image_path):
            os.makedirs(image_path)

        clip_path += f'\\{dir}.mp4'

        ffmpeg_command = ffmpeg_path + ' -i ' + clip_path + " " + image_path + '\\frame_%06d.png'
        print(ffmpeg_command)
        os.system(ffmpeg_command)
        
createFrames()
