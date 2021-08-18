from genericpath import exists
import json
import os
import shutil

def processAnnotations(input_path=r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new'):
    # coco_format is the dict file which includes all the values that needs to be output in the final annotations json file
    # Some of the key values like 'licenses', 'info' and 'categories' are constant and declared at first here

    coco_format = {
        "licenses": [{
            "name": "",
            "id": 0,
            "url": ""
        }],
        "info": {
            "contributor": "Vishal Balaji",
            "date_created": "",
            "description": "Escooter Dataset",
            "url": "",
            "version": "",
            "year": ""
        },
        "categories": [{
            "id": 1,
            "name": "Escooter",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "Pedestrian",
            "supercategory": ""
        },
        {
            "id": 3,
            "name": "Cyclist",
            "supercategory": ""
        }]
    }

    # The key values 'images' and 'annotations' needs to be processed and appended. The below given lines is the format for
    # those dicts.
    """
    "images":[
        {
            "id":1,
            "width": 1920,
            "height": 1080,
            "file_name":"sdfa.PNG",
            "license":0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
    ]

    "annotations":[
        {
            "id": 1,
            "image_id": 55,
            "category_id": 1,
            "segmentation": [[]],
            "area": {some area number in float},
            "bbox": [].
            "iscrowd": 0
        }
    ]
    """

    # Path where the annotations are stored, when the repo is the path of current working directory
    #main_file_path = os.path.abspath(r'D:\Carissma Video Copy\Traffic Camera Tracking\Finished')
    #input_path = r'C:\Vishal-Videos\Project_Escooter_Tracking\input'
    main_file_path = input_path

    # Declaration of empty lists that is later appended it with images and annotations.
    images_list_with_Anno = []
    annotations_list = []

    # Each image and annotations has an ID associated with it and it starts with 1.
    # These values are incremented as the images and annotations are being added.
    img_num = 1
    anno_num = 1

    escooter_count = 0 
    pedestrian_count = 0
    cyclist_count = 0

    def adjustFrameDifference(file_name, offset=1):
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

    print("- Processing the following annotation files: ")
    for clips in os.listdir(main_file_path):
        # Checking that only numbers are given as folder names for the clips
        if all(char.isdigit() for char in clips):
            # Path of the clips folder
            clips_path = main_file_path + '\\' + clips
            # Path of the annotation of the clips
            
            annotation_file = clips_path + f'\\COCO\\annotations\instances_default.json'
            
            file = open(annotation_file)
            json_file = json.load(file)
            print(f'  - {annotation_file}')
                        
            # !! Testing purpose only for restricting number of annotations
            # flag = 1
            for annotations in json_file['annotations']:

                anno_image_ID = annotations['image_id']
                anno_ID = annotations['id']

                image_filename = ''
                for images in json_file['images']:
                    if images['id'] == anno_image_ID:
                        image_filename = images['file_name']

                filename = input_path + '\\' + clips + '\\images\\' + image_filename
                filename = adjustFrameDifference(filename)  
                
                # The formats for 'images' dictionary and 'annotations' dictionary in COCO
                image_dict = {
                    'id': img_num,
                    "width": 1920,
                    "height": 1080,
                    "file_name": filename,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                }
                anno_dict = {
                    "id": anno_num,
                    'image_id': img_num,
                    "category_id": annotations['category_id'],
                    'segmentation': [],
                    'area': annotations['area'],
                    'bbox': annotations['bbox'],
                    'iscrowd': annotations['iscrowd']
                }

                # In the COCO-Format, every images and associated annotations are passed as array of dicts.
                images_list_with_Anno.append(image_dict)
                annotations_list.append(anno_dict)

                #imgNumWithAnnotations.append(img_num)
                # Incrementing the Image ID and Annotation ID for each loop
                img_num += 1
                anno_num += 1

                if annotations['category_id'] == 1:
                    escooter_count += 1
                elif annotations['category_id'] == 2:
                    pedestrian_count += 1
                elif annotations['category_id'] == 3:
                    cyclist_count += 1

                
            file.close()
        
                    
            # !! Meant for testing purpose.
            # if clip_number == 1:
            #     break

    print(f'\n- Total no.of annotations in the dataset: {anno_num}')

    print(f'\n- Total no.of escooter instances: {escooter_count}')
    print(f'\n- Total no.of pedestrians instances: {pedestrian_count}')
    print(f'\n- Total no.of cyclist instances: {cyclist_count}')
    
    train_json = deepcopy(coco_format)
    valid_json = deepcopy(coco_format)
    test_json = deepcopy(coco_format)

    train_split = 0.85
    valid_split = 0.15
    #test_split = 0.1

    # Function to split the whole dataset of images and annotations into train,
    # valid and test sets
    def splitDataset(img_Anno, annotations, trainSplit):
        
        trainSize = int(len(img_Anno) * trainSplit)
        train_images = []
        train_annotations = []
        
        valid_images = list(img_Anno)
        valid_annotations = list(annotations)
        while len(train_images) < trainSize:            
            index = random.randrange(len(valid_images))           
            train_images.append(valid_images.pop(index))
            train_annotations.append(valid_annotations.pop(index))
        
        # copySize = int(len(copy_images) * (validSplit/(1 - trainSplit)))
        # valid_images = []
        # valid_annotations = []

        # test_images = copy_images
        # test_annotations = copy_annotations
        # while len(valid_images) < copySize:
        #     index = random.randrange(len(test_images))
        #     valid_images.append(test_images.pop(index))
        #     valid_annotations.append(test_annotations.pop(index))

        
        
        # return [(train_images, train_annotations), (valid_images, valid_annotations), (test_images, test_annotations)]
        return [(train_images, train_annotations), (valid_images, valid_annotations)]

    #train_set, valid_set, test_set = splitDataset(images_list_with_Anno, images_list_without_Anno, annotations_list, 0.8, 0.1)
    train_set, valid_set = splitDataset(images_list_with_Anno, annotations_list, trainSplit=train_split)
    print("\n- Splitting the dataset into Train, Valid and Test is successfull\n")

    # Storing the processed arrays of images and annotations with their
    # respective keys in the final dataset
    # coco_format["images"] = images_list
    # coco_format["annotations"] = annotations_list

    train_json['images'] = train_set[0]
    train_json['annotations'] = train_set[1]

    valid_json['images'] = valid_set[0]
    valid_json['annotations'] = valid_set[1]

    # test_json['images'] = test_set[0]
    # test_json['annotations'] = test_set[1]

    # Code Snippet to automatically create new names for the many
    # .json files created during the testing
    base_filename = 'Test_'
    for numbers in range(20):
        check_filename = base_filename + str(numbers+1) + '.json'
        if check_filename not in os.listdir(os.getcwd()):
            base_filename = check_filename
            break


    # These lines writes all the dictionaries into the final required .json file
    # For train, valid and test individually
    train_file = f"{input_path}\\{base_filename[:-5]}_Train.json"
    valid_file = f"{input_path}\\{base_filename[:-5]}_Valid.json"
    # test_file = f"{input_path}\\{base_filename[:-5]}_Test.json"

    print("- Saving train, test and valid annotation files")
    with open(train_file, "w") as file:
        json.dump(train_json, file)
        print(f"  - Final training set file saved as: {train_file}")

    with open(valid_file, "w") as file:
        json.dump(valid_json, file)
        print(f"  - Final valid set file saved as: {valid_file}")

    # with open(test_file, "w") as file:
    #     json.dump(test_json, file)
    #     print(f"  - Final test set file saved as: {test_file}")

    # loadDataset(train_file, valid_file, test_file)
    loadDataset(train_file, valid_file)
    
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
    

def main():
    main_path = os.path.abspath(r'C:\Users\balaji\Desktop\Yolo')
    init_json_path = os.path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\YoloV5\COCO2YOLO\Test_JSONS\Test_1_Valid.json')
    
    main_dict = json.load(open(init_json_path))
    
    # Creating a custom dictionary for keeping track of id's and the associated (label name) name for writing in the obj's files
    classes = []
    for items in main_dict['categories']:
        new_class = {'id': items['id'], 'name': items['name']}
        classes.append(new_class)

    # Creating Yolo data folder
    yolo_folder = f'{main_path}\Yolo_data'
    os.makedirs(yolo_folder, exist_ok=True)
    create_Objs(yolo_folder, classes)
    
    # for files in os.listdir(main_path):
    #     clip_folder = main_path + '\\' + files
    #     frame_number = 0
    #     if os.path.isdir(clip_folder):
    #         annotation_path = clip_folder + '\\COCO\\annotations\\instances_default.json'
    #         annotation_dict = json.load(open(annotation_path))
    #         #print(annotation_dict.keys())

    #         for image in annotation_dict['images']:
    #             print(image.keys())
    #             count = 0
    #             for anno in annotation_dict['annotations']:
    #                 print(anno.keys())
    #                 if image['id'] == anno['image_id']:
    #                     print(anno['bbox'])
    #                 break
    #             break
    #     break

    # Creating train.txt, valid.txt and data directory
    # This deletes any pre-existing files and creating new ones.
    train = open(f'{yolo_folder}\\train.txt', 'w+')
    valid = open(f'{yolo_folder}\\valid.txt', 'w+')
    data_directory = f'{yolo_folder}\\data'
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
    os.makedirs(data_directory)

    print(main_dict.keys())

    img_count = 0
    for image in main_dict['images']:
        # Copying and renaming the files
        shutil.copy2(image['file_name'], data_directory)
        new_img_name = f'frame_{img_count}.png'
        os.rename(f"{data_directory}\\{image['file_name'][-16:]}", f"{data_directory}\\{new_img_name}")
        valid.write(f'{data_directory}\\{new_img_name}\n')

        frame_txt = open(f'{data_directory}\\frame_{img_count}.txt', 'w+')
        anno_string = ''
        for anno in main_dict['annotations']:
            if image['id'] == anno['image_id']:
                #print(f'Annotations: {anno}\nImage: {image}')
                yolo_bbox = convert_coco_bbox_to_yolo(anno['bbox'], image['width'], image['height'])
                anno_string += f"{anno['category_id'] - 1} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"
        if anno_string != '':
            frame_txt.write(anno_string[:-1])
        img_count += 1
        frame_txt.close()

    train.close()
    valid.close()
        
def check():
    main_path = os.path.abspath(r'C:\Users\balaji\Desktop\Yolo')
    init_json_path = os.path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\YoloV5\COCO2YOLO\Test_JSONS\42.json')
    
    main_dict = json.load(open(init_json_path))
    for image in main_dict['images']:
        count = 0
        list_anno = []
        for anno in main_dict['annotations']:
            if image['id'] == anno['image_id']:
                count += 1
                list_anno.append(anno)
        #print(f"Filename: {image['file_name']}, Count of Annotations: {count}")
        if count == 2:
            print(list_anno)
            break

#main()
check()