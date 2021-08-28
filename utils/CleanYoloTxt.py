"""
Simple Script to clean annotation file stored in the input data directory while testing with Yolo Annotation Format.
Can be reused in the project in the future to delete files of certain format from a directory.
"""
import os

main_path = os.path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new')

for files in os.listdir(main_path):
    files_path = f'{main_path}\\{files}'
    if os.path.isdir(files_path):
        print(files_path)
        images_path = f'{files_path}\\images'
        for _ in os.listdir(images_path):
            if _[-4:] == '.txt':
                txt_file_path = f'{images_path}\\{_}'
                os.remove(txt_file_path)
                print(f'Removing {txt_file_path}')

