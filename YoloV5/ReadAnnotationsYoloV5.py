import random
import json
import os
import shutil

def create_Objs(yolo_data_folder, class_dict):
    num_class = 0

    # Creating Obj.names file
    obj_names_path = yolo_data_folder + f'\\obj.names'
    if os.path.exists(obj_names_path):
        os.remove(obj_names_path)
    class_str = ''
    with open(obj_names_path, 'a') as f:
        for items in class_dict:
            class_str += items['name'] + '\n'
            num_class += 1
        f.write(class_str[:-1])
            
    
    # Creating Obj.data file
    obj_data_path = yolo_data_folder + f'\\obj.data'
    if os.path.exists(obj_data_path):
        os.remove(obj_data_path)
    with open(obj_data_path, 'a') as f:
        f.write(f'classes = {num_class}\n')
        f.write(f'train = {yolo_data_folder}\\train.txt\n')
        f.write(f'valid = {yolo_data_folder}\\valid.txt\n')
        f.write(f'names = {obj_names_path}\n')
        f.write(f'backup = backup/')

def convert_coco_bbox_to_yolo(bbox, im_width, im_height):
    """
    Function for converting COCO format BBOX to Yolo format

    COCO Format: x1, y1, width, height              
    => where x1, y1 represents the top left corner of bbox

    YOLO Format: x1, y1, total_width, total_height  
    => where x1, y1 represent the center of the bbox
    => All variables are scaled according to the resolution. So their values can only be between 0 and 1
    """

    center_x = ((2 * bbox[0]) + bbox[2])/ (2 * im_width)
    center_y = ((2 * bbox[1]) + bbox[3])/ (2 * im_height)
    total_width = bbox[2] / im_width
    total_height = bbox[3] / im_height

    return (center_x, center_y, total_width, total_height)

def adjustFrameDifference(file_name, offset=1):
        """
        In CVAT, the frames are numbered starting from 0.
        But while splitting the video into individual frames by FFMPEG, the frame numbering starts from 1.
        This function takes care of that difference.
        """
    
        # Adjusting for difference in frame
        file_name_from_dict = file_name.split('.')[0]
        file_number = int(file_name_from_dict[-6:])
        
        # 1 is the offset number for the frame difference between the annotations 
        # from CVAT and frames extracted from the FFMPEG Script
        file_number += 1
        
        # Adding the write number of 0's and taking care of proper filename
        if int(file_number / 10) == 0:
            new_file_name = file_name_from_dict[:-6] + '00000' + str(file_number) + '.png'
        elif int(file_number / 100) == 0:
            new_file_name = file_name_from_dict[:-6] + '0000' + str(file_number) + '.png'
        elif int(file_number / 1000) == 0:
            new_file_name = file_name_from_dict[:-6] + '000' + str(file_number) + '.png'
        elif int(file_number / 10000) == 0:
            new_file_name = file_name_from_dict[:-6] + '00' + str(file_number) + '.png'
        
        return new_file_name

def main(main_path, files_path, train_split):
    # Creating Yolo Folder (train.txt, valid.txt and data directory)
    # This deletes any pre-existing files and creating new ones.

    yolo_folder = f'{main_path}\Yolo_data'
    os.makedirs(yolo_folder, exist_ok=True)

    train = open(f'{yolo_folder}\\train.txt', 'w+')
    valid = open(f'{yolo_folder}\\valid.txt', 'w+')
    
    data_directory = f'{yolo_folder}\\data'
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
    os.makedirs(data_directory)

    # Variable used to check that obj.names and obj.data files are created once
    is_base_files_created = False
    # Frame Number which increments for every frame read and copied
    frame_number = 0

    # To count different conditions, which are thenprinted in a short report at the end
    total_clips = 0
    total_images = 0
    total_instances = 0
    total_escooter_instances = 0
    total_pedestrian_instances = 0
    total_cyclist_instances = 0

    all_images = []
    for files in os.listdir(files_path):
        # Looping through each folder and getting the annotations
        clip_folder = files_path + '\\' + files

        if os.path.isdir(clip_folder):
            total_clips += 1
            annotation_path = clip_folder + '\\COCO\\annotations\\instances_default.json'
            annotation_dict = json.load(open(annotation_path))
            
            print(f'Annotations Path: {annotation_path}')
            if not is_base_files_created:
                # Creating a custom dictionary for keeping track of id's and the associated (label name) name for writing in the obj's files
                classes = []
                for items in annotation_dict['categories']:
                    new_class = {'id': items['id'], 'name': items['name']}
                    classes.append(new_class)
                create_Objs(yolo_folder, classes)

                is_base_files_created = True

            escooter_instances = 0
            pedestrian_instances = 0
            cyclist_instances = 0

            for image in annotation_dict['images']:
                # Converting the relative path in each annotation to abs path and adjusting for frame offset
                abs_img_path = clip_folder + f"\\images\\{image['file_name']}"
                frame_adjusted_img_filename = adjustFrameDifference(abs_img_path)

                # Copying and renaming the files
                shutil.copy2(frame_adjusted_img_filename, data_directory)
                new_img_name = f'frame_{frame_number}.png'
                os.rename(f"{data_directory}\\{frame_adjusted_img_filename[-16:]}", f"{data_directory}\\{new_img_name}")
                #valid.write(f'{data_directory}\\{new_img_name}\n')
                all_images.append(f'{data_directory}\\{new_img_name}\n')

                frame_txt = open(f'{data_directory}\\frame_{frame_number}.txt', 'w+')
                anno_string = ''
                
                for anno in annotation_dict['annotations']:
                    if image['id'] == anno['image_id']:
                        yolo_bbox = convert_coco_bbox_to_yolo(anno['bbox'], image['width'], image['height'])
                        anno_string += f"{anno['category_id'] - 1} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"
                        
                        if anno['category_id'] == 1:
                            escooter_instances += 1
                        elif anno['category_id'] == 2:
                            pedestrian_instances += 1
                        elif anno['category_id'] == 3:
                            cyclist_instances += 1
                
                total_images += 1
        
                if anno_string != '':
                    frame_txt.write(anno_string[:-1])
                frame_number += 1
                frame_txt.close()
            
            clip_instances = escooter_instances + pedestrian_instances + cyclist_instances
            total_instances += clip_instances
            total_escooter_instances += escooter_instances
            total_pedestrian_instances += pedestrian_instances
            total_cyclist_instances += cyclist_instances

            print(f"Finished Processing annotations for the clip number: {clip_folder}")
            print(f"Total Instances in the clip: {clip_instances}")
            print(f"Escooter Instances: {escooter_instances}")
            print(f"Pedestrian Instances: {pedestrian_instances}")
            print(f"Cyclist Instances: {cyclist_instances}\n")          

    print(f"Successfully finished processing {total_clips} clips.\n\nREPORT:\n")
    print(f"Total number of Images: {total_images}")
    print(f"Total Instances: {total_instances}")
    print(f"Total Escooter Instances: {total_escooter_instances}")
    print(f"Total Pedestrian Instances: {total_pedestrian_instances}")
    print(f"Total Cyclist Instances: {total_cyclist_instances}")
    
    random.shuffle(all_images)
    train_data = all_images[:int((len(all_images)+1)*train_split)]
    validation_data = all_images[int((len(all_images)+1)*train_split):]

    train.writelines(train_data)
    valid.writelines(validation_data)

    train.close()
    valid.close()

output_path = os.path.abspath(r'C:\Users\balaji\Desktop\Yolo')
clips_path = os.path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new')
train_valid_split = 0.85    # 85% is in train and 15% is in validation

main(output_path, clips_path, train_valid_split)