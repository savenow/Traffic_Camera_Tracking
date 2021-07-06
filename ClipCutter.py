from moviepy.editor import VideoFileClip, CompositeVideoClip
from os import path, listdir, mkdir, system

# Linux File Paths. Uncomment while using in the linux computer
# recordings_path = path.abspath('/home/escooter/Videos/ffmpeg_cron/40_minutes')
# code_path = path.abspath('/home/escooter/Videos/CodeFiles')
# output_path = path.abspath('/home/escooter/Videos/CodeFiles/Output_clips')

# Windows File Paths. Uncomment while using in the windows machine.
recordings_path = path.abspath(r'C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Recordings')
code_path = path.abspath(r'C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Main_Code\Traffic_Camera_Tracking')
output_path = path.abspath(r'C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Main_Code\RPC_processed\Output_clips\Test')

def findFileNumber():
    """
    This function is meant to find the number in the existing clips and start new file number from the last clip
    present in the directory
    """
    file_number = []
    for fileName in listdir(output_path):
        fileName_number, *_ = fileName.split('.')
        file_number.append(int(fileName_number))
    if not file_number:
        OUTPUT_FILE_NUMBER = 1
    else:
        OUTPUT_FILE_NUMBER = max(file_number) + 1
    return OUTPUT_FILE_NUMBER

def createFrames(file_name, start_time, end_time, outputNumber):
    video_file_path = output_path + '\\' + str(outputNumber) + '\\' + str(outputNumber) + '.mp4'
    # Change in linux to '/'
    output_directory_images = output_path + '\\' + str(outputNumber) + '\\images'
    mkdir(output_directory_images)

    #print(len(start_time[0]), len(end_time[0]))
    if len(start_time[0]) == 1:
        start_time[0] = '0' + start_time[0]
        print('\nStart Time If Block')
        print(start_time[0])
    if len(end_time[0]) == 1:
        end_time[0] = '0' + end_time[0]
        print('\nEnd Time If Block')
        print(end_time[0])

    """
    ffmpeg_command = 'ffmpeg -i ' + str(video_file_path) + ' -qscale:v 6 -ss 00:' + str(
        int(start_time / 100)) + ':' + str(int(start_time - int(start_time / 100))) + ' -t 00:' + str(
        int(end_time / 100)) + ':' + str(
        int(end_time - int(end_time / 100))) + ' ' + output_directory_images + '/frame_%06d.png'
    ffmpeg_command_1 = 'ffmpeg -i ' + str(video_file_path) + ' -qscale:v 6 -ss ' + str(start_time) + ' -t ' + str(
        end_time) + ' ' + output_directory_images + '/frame_%06d.png'
    ffmpeg_command_2 = 'ffmpeg -i ' + str(video_file_path) + ' -ss 00:' + start_time[0] + ':' + start_time[1] \
                       + ' -t 00:' + end_time[0] + ':' + end_time[1] + ' ' + output_directory_images + '\\frame_%06d.png'
    """


    # """ffmpeg -i C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Main_Code\RPC_processed\Output_clips\2.mp4
    # -qscale:v 6 C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Main_Code\RPC_processed\Output_clips\2\frame_%06d.png"""

    ffmpeg_command = 'ffmpeg -i ' + str(video_file_path) + ' ' + output_directory_images + '\\frame_%06d.png'
    print(ffmpeg_command)
    system(ffmpeg_command)

def clipvideo(file_name, start_time, end_time, outputNumber):
    """
    Clipping the video file from required timestamps and outputting the clips in the required output file numbers.
    """
    video_file_path = path.join(recordings_path, file_name)
    output_directory_video = output_path + '\\' + str(outputNumber)
    mkdir(output_directory_video)

    clip = VideoFileClip(video_file_path).subclip(start_time, end_time).without_audio()
    output_clipname = str(outputNumber) + '.mp4'
    output_clip_path = path.join(output_directory_video, output_clipname)
    clip.write_videofile(output_clip_path, codec='libx264')

def clipToImages(file_name, start_time, end_time, outputNumber):
    """
    Clipping the video file from required timestamps and outputting the image sequence in the required output file numbers.
    """
    video_file_path = path.join(recordings_path, file_name)
    output_directory_video = output_path + '\\' + str(outputNumber)
    mkdir(output_directory_video)

    clip = VideoFileClip(video_file_path).subclip(start_time, end_time).without_audio()
    output_clipname = 'frame_%06d.png'
    output_clip_path = path.join(output_directory_video, output_clipname)
    clip.write_images_sequence(output_clip_path)

def read_file(path, outputFileNumber):
    with open(path) as f:
        file_name = ''
        start_time = 0
        end_time = 0
        for line in f:
            #print(line)
            if line[:2] == 'VF':
                file_name = line[4:-1]
                # -1 becoz the last character is newline character \n
                print(file_name)
            else:
                start_timestamp, *end_timestamp = line.split(sep=' - ')
                start_timestamp = start_timestamp.split(sep=':')
                start_time = int(start_timestamp[0]) * 60 + int(start_timestamp[1])
                if not end_timestamp:
                    end_time = start_time + 15
                else:
                    end_timestamp = str(end_timestamp[0]).split(sep=':')
                    end_time = int(end_timestamp[0]) * 60 + int(end_timestamp[1])
                print(start_time, end_time)
                clipvideo(file_name, start_time, end_time, outputFileNumber)
                #clipToImages(file_name, start_time, end_time, outputFileNumber)
                createFrames(file_name, start_timestamp, end_timestamp, outputFileNumber)
                outputFileNumber += 1


def main():
    output_file_number = findFileNumber()

    # Windows File Path - Local PC
    text_file_path = path.join(code_path, 'Recordings_Windows.txt')
    # Linux File Path - Remote PC
    # text_file_path = path.join(code_path, 'Recordings_Linux.txt')
    read_file(text_file_path, output_file_number)

main()
