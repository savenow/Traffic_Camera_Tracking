import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A

#segmentation = [[375.2, 826.88, 374.9, 825.24, 374.86, 823.9, 375.72, 822.64, 376.9, 821.44, 378.1, 820.32, 378.5, 818.84, 378.18, 817.46, 377.54, 816.08, 376.9, 815.04, 376.32, 813.46, 375.56, 811.96, 374.86, 810.62, 374.02, 809.3, 372.86, 808.18, 371.54, 807.82, 369.7, 807.7, 368.36, 807.02, 366.98, 806.2, 365.68, 805.28, 364.82, 803.76, 364.68, 802.18, 364.38, 800.66, 363.9, 799.46, 363.4, 797.94, 362.82, 796.44, 362.48, 794.86, 362.54, 793.2, 363.0, 791.7, 363.82, 790.36, 364.56, 788.9, 365.16, 787.38, 365.76, 785.72, 366.86, 784.54, 368.2, 783.6, 369.7, 783.14, 371.14, 782.3, 372.66, 782.08, 374.16, 782.4, 375.44, 783.26, 376.38, 784.28, 377.34, 785.34, 378.74, 786.14, 380.24, 786.44, 381.82, 787.0, 383.32, 787.68, 384.52, 788.0, 386.12, 788.44, 387.76, 788.64, 389.22, 788.9, 390.42, 789.68, 390.86, 791.06, 390.54, 792.52, 389.94, 793.56, 391.2, 794.58, 392.34, 795.8, 393.52, 796.86, 394.34, 798.16, 395.38, 799.32, 396.42, 800.5, 397.84, 801.06, 399.04, 801.22, 400.48, 801.6, 401.86, 802.26, 402.7, 803.54, 402.9, 805.04, 402.62, 806.74, 401.62, 808.16, 400.0, 808.56, 398.5, 808.76, 397.28, 809.68, 397.02, 811.26, 397.02, 812.76, 396.26, 814.08, 395.06, 815.18, 393.84, 816.14, 393.04, 817.44, 392.02, 818.8, 390.86, 819.92, 389.98, 820.8, 389.1, 822.06, 388.02, 823.42, 386.88, 824.64, 385.82, 825.76, 384.82, 827.08, 383.56, 828.04, 382.16, 828.7, 380.66, 829.08, 379.14, 828.86, 377.62, 828.54, 376.3, 827.68]]
segmentation = [[377.65, 824.24, 377.4, 822.57, 377.38, 821.25, 378.36, 820.02, 379.55, 818.82, 380.75, 817.71, 381.05, 816.17, 380.59, 814.78, 379.87, 813.44, 379.15, 812.27, 378.56, 810.68, 377.78, 809.18, 377.03, 807.86, 376.21, 806.55, 374.98, 805.49, 373.57, 805.26, 371.7, 805.15, 370.38, 804.41, 369.04, 803.6, 367.74, 802.69, 366.86, 801.18, 366.74, 799.59, 366.39, 798.08, 365.85, 796.73, 365.35, 795.22, 364.76, 793.72, 364.39, 792.13, 364.47, 790.45, 365.0, 788.95, 365.91, 787.63, 366.68, 786.2, 367.28, 784.69, 367.88, 783.01, 368.98, 781.82, 370.35, 780.9, 371.85, 780.42, 373.27, 779.45, 374.78, 779.14, 376.28, 779.45, 377.52, 780.38, 378.39, 781.54, 379.27, 782.72, 380.62, 783.57, 382.12, 783.87, 383.71, 784.45, 385.21, 785.14, 386.56, 785.5, 388.16, 785.92, 389.83, 786.07, 391.31, 786.25, 392.61, 786.94, 393.23, 788.33, 392.77, 789.81, 391.92, 790.88, 393.2, 791.89, 394.37, 793.1, 395.56, 794.13, 396.37, 795.43, 397.34, 796.66, 398.31, 797.95, 399.72, 798.53, 401.07, 798.71, 402.54, 799.0, 403.98, 799.58, 404.95, 800.82, 405.25, 802.32, 405.06, 804.07, 404.11, 805.58, 402.5, 805.98, 401.0, 806.18, 399.74, 807.04, 399.46, 808.63, 399.51, 810.13, 398.73, 811.44, 397.53, 812.54, 396.32, 813.52, 395.52, 814.82, 394.51, 816.2, 393.38, 817.31, 392.39, 818.3, 391.55, 819.58, 390.46, 820.96, 389.34, 822.22, 388.31, 823.38, 387.31, 824.69, 386.03, 825.62, 384.63, 826.25, 383.13, 826.59, 381.62, 826.33, 380.11, 825.97, 378.8, 825.09]]

bbox = [364.39, 779.14, 40.86, 47.45]

pts = []


def visualizeMasks(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 3, figsize=(8, 8))

        ax[0].imshow(image)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(mask)
        ax[1].set_title('Original Mask', fontsize=fontsize)

        original_annotation = cv2.bitwise_and(image, mask)
        ax[2].imshow(original_annotation)
        ax[2].set_title('Annotated Image', fontsize=fontsize)

        print(f'Image Shape: {image.shape}')
        print(f'Mask Shape: {mask.shape}')
    else:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        f, ax = plt.subplots(3, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        original_annotation = cv2.bitwise_and(original_image, original_mask)
        ax[2, 0].imshow(original_annotation)
        ax[2, 0].set_title('Original Annotation', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

        masked_annotation = cv2.bitwise_and(image, mask)
        ax[2, 1].imshow(masked_annotation)
        ax[2, 1].set_title('Transformed Annotation', fontsize=fontsize)

        print(f'Image Shape: {image.shape}')
        print(f'Mask Shape: {mask.shape}')

    plt.show()


def createMask(img, seg_Array, debug=False):

    # The points received are in serial order where [n, n+1..] element represents (x, y) points
    # This code converts this list into a list of (x, y) points
    index = 0
    while index < len(seg_Array):
        pts.append((seg_Array[index], seg_Array[index+1]))
        index += 2

    # For cv2.fillPoly function, the input points must be an array of INTEGER 32.
    points = np.array(pts, np.int32)
    if debug:
        print('The Points are: ')
        print(points)

    blank_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.fillPoly(blank_image, [points], (255, 255, 255), lineType=cv2.LINE_AA)

    if debug:
        cv2.namedWindow('Mask_Image')
        cv2.imshow('Mask_Image', blank_image)
        cv2.waitKey(0)

    return blank_image


def main():
    image = cv2.imread(r'C:\Users\visha\Desktop\Carissma\TrafficMonitoring\Main_Code\Traffic_Camera_Tracking\Sample_Pictures\4_Annotated.png')
    mask = createMask(image, segmentation[0])

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

    augmented = transform(image=image, mask=mask)

    image_h_flipped = augmented['image']
    mask_h_flipped = augmented['mask']
    visualizeMasks(image_h_flipped, mask_h_flipped, image, mask)
    #visualizeMasks(image, mask)

main()



