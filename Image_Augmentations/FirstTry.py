import random
import cv2
from matplotlib import pyplot as plt
import os
import albumentations as A


def visualize(orig_image, transformed_image):
    f, axis = plt.subplots(1, 2)
    axis[0].imshow(orig_image)
    axis[1].imshow(transformed_image)
    plt.show()


def printProperties(data):
    print(f'Property Name: {data["__class_fullname__"]}')
    for key_values in data.keys():
        if key_values != '__class_fullname__':
            print(f"\t- {key_values}: {data[key_values]}")


def main():
    image = cv2.imread(r'C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Main_Code\Traffic_Camera_Tracking\Sample_Pictures\3.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # transform = A.OneOf([
    #     #A.IAAAdditiveGaussianNoise(),
    #     A.GaussNoise()
    # ], p=0.5)

    # Transformations included:
    # 1. A.Oneof([A.GaussNoise(), A.Blur(blur_limit=3), A.MedianBlur(blur_limit=4)], p=0.35)
    # 2. A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30)
    # 3. A.OpticalDistortion(p=0.6)
    transform = A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussNoise(),
            A.Blur(blur_limit=3),
            A.MedianBlur(blur_limit=5)
        ], p=0.35),
        A.Compose([
            A.RandomCrop(width=1280, height=720, p=0.7),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=0.6),
            A.RGBShift(p=0.5),
        ], p=0.5),
    ])
    transform_1 = A.ReplayCompose([
        A.SafeRotate(limit=10, p=1),
        A.HorizontalFlip(p=0),
    ])

    images_path = os.path.abspath(
        r'C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Main_Code\Traffic_Camera_Tracking\Image_Augmentations\Trial_Images')
    #random.seed(7)
    count = 0
    while count < 11:
        augmented_data = transform(image=image)
        augmented_image = augmented_data['image']
        #print(augmented_data['replay']['transforms'])

        print(f"Image Number: {count+1}")

        for transform_objs in augmented_data['replay']['transforms']:
            if transform_objs['__class_fullname__'] == 'OneOf':
                for more_transforms in transform_objs['transforms']:
                    if more_transforms['applied'] is True:
                        printProperties(more_transforms)
            elif transform_objs['applied'] is True:
                printProperties(transform_objs)

        file_path = images_path + f'\\{count+1}.png'
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file_path, augmented_image)
        #visualize(image, augmented_image)

        count += 1

main()