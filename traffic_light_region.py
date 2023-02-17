import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
import pickle
import pandas as pd
import numpy as np
import colorsys
import time

# Region of Interest
class ROI:
    def __init__(self, input, region, night_mode, output_dir):
        self.region = region
        self.mean = 0
        self.fps = 30
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.state_list = ['Green', 'Red', 'Blue', 'Yellow']
        self.night_mode = night_mode
        self.not_detected_path = './not detected'

        if output_dir == None:
            self.output_dir_path = './data'
            self.not_detected_path = './not detected'
        else:
            self.output_dir_path = output_dir + '/data'
            self.not_detected_path = output_dir + '/not detected'

        # Checking input
        if os.path.isfile(input):
            # Further functionality needs to be added for Folder Inference :))
            if input[-4:] in ['.png', '.jpg']:
               self.input = cv2.imread(input)
               self.target_mode = 'Image'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = cv2.VideoCapture(input)
                self.target_mode = 'Video'
            else:
                print("Invalid input file. The file should be an image or a video !!")
                exit(-1)
        else:
            print("Input file doesn't exist. Check the input path")
            exit(-1)

        # Assigning ROI and color threshold based on night or day video mode
        if self.night_mode:
            self.red_region = region[2]
            self.green_region = region[3]
            self.green_params = {'hue_low': 25, 'hue_high': 40, 
                                 'sat_low': 40, 'sat_high': 100, 
                                 'val_low': 190, 'val_high':255}

            self.red_params = {'hue_low': 90, 'hue_high': 120, 
                               'sat_low': 0, 'sat_high': 58, 
                               'val_low': 230, 'val_high':255}
        else:
            self.red_region = region[0]
            self.green_region = region[1]
            self.green_params = {'hue_low': 25, 'hue_high': 40, 
                                 'sat_low': 113, 'sat_high': 255,
                                 'val_low': 170, 'val_high':255}

            self.red_params = {'hue_low': 90, 'hue_high': 120, 
                               'sat_low': 123, 'sat_high': 255, 
                               'val_low': 193, 'val_high':255}

        # Creating Output Directory
        if not os.path.exists(self.output_dir_path):
                os.makedirs(self.output_dir_path)
        else:
            shutil.rmtree(self.output_dir_path)       # Removes all the subdirectories!
            os.makedirs(self.output_dir_path)

        # Creating 'not detected' Directory to store undetected frames
        if not os.path.exists(self.not_detected_path):
                os.makedirs(self.not_detected_path)
        else:
            shutil.rmtree(self.not_detected_path)       # Removes all the subdirectories!
            os.makedirs(self.not_detected_path)

        self.csv_data_final = open(self.output_dir_path + '/state_data_final.csv', 'a')

        # Run Main method
        # self.run()
    
    def region_of_interest(self, img, dims):
        '''
        Crop input images
        Returns region of interest image
        '''
        cropped_img =  img[dims[0]:dims[2], dims[1]:dims[3]]
        return cropped_img

    def load_data(self, dir_name = 'images'):    
        '''
        Load images from the "faces_imgs" directory
        Images are in JPG and we convert it to gray scale images
        '''
        imgs = []
        for filename in os.listdir(dir_name):
            img = mpimg.imread(dir_name + '/' + filename)
            #img = skimage.color.rgb2gray(img)
            imgs.append(img)
        return imgs

    def rgb2hsv(self, rgb):
        '''
        takes RGB values as args and converts it into HSV form. HSV form is easy to define a color, intensity and grayness of color.
        returns the list of Three Values. e.g [123, 98, 74]
        Note: Hue Range: 0° to 360°, Saturation: from 0% to 100%, Values: from 0% to 100%
        '''
        params = []
        R, G, B = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
        hsv = colorsys.rgb_to_hsv(R, G, B)
        hsv_list = list(i for i in hsv)
        params.extend(hsv_list)
        new_params = self.range_conversion(hsv_list)
        return new_params
    
    def RGB_list(self, img):
        ''' Takes Image input and creates list of RGB values'''
        dir = []
        for i in img:
            for j in i:
                dir.append(j.tolist())
        return dir
        
    def HSV_list(self, rgb):
        '''Takes RGB list as input and creates HSV List using func. rgb2hsv'''
        hsv_list = []
        for i in rgb:
            hsv_list.append(self.rgb2hsv(i))
        return hsv_list

    def pixel_change(self, img, hsv_list):
        '''Takes image and HSV list as input. It transforms RGB image into HSV image by replacing pixel values'''
        hsv_list_counter = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # print(hsv_list_counter)
                img[i, j] = (int(hsv_list[hsv_list_counter][0]), int(hsv_list[hsv_list_counter][1]), int(hsv_list[hsv_list_counter][2]))
                hsv_list_counter += 1
        return img

    def range_conversion(self, color_list):
        """
        Takes HSV color list as input
        Return: HSV list with different color range!
        eg. H: 0° - 360° --> H: 0° - 180°
            S: 0° - 100° --> S: 0 - 255
            V: 0° - 100° --> V: 0 - 255
        """
        new_hsv = []
        H = color_list[0] * 180
        S = color_list[1] * 255
        V = color_list[2] * 255

        new_hsv.extend([H, S, V])

        return new_hsv

    def frame_visualizer(self, imgs, format=None, gray=False):
        '''
        Display Images from the argument list.
        It displays max 4 images on a window.
        '''
        for i, img in enumerate(imgs):
            if img.shape[0] == 3:
                img = img.transpose(1,2,0)
            plt_idx = i+1
            plt.subplot(2, 2, plt_idx)
            # print(img)
            plt.imshow(img, format)
        plt.show()

    def light_state(self, frame):
        rgb_list_red = self.RGB_list(frame[0])
        rgb_list_green = self.RGB_list(frame[1])
        
        hsv_list_red = self.HSV_list(rgb_list_red)
        hsv_list_green = self.HSV_list(rgb_list_green)

        # Mean of Hue values 
        hue_sum_red = 0     # Sum of all Hue values which satisfies the condition for RED Region
        hue_sum_green = 0   # Sum of all Hue values which satisfies the condition for GREEN Region
        length_green = 0    # Length of Green value
        length_red = 0      # Length of Red color
        
        # Intensity of Saturation and Value
        sat_on_pixels_red = 0     # number of Saturation Pixels that are Bright in Red Region
        sat_on_pixels_green = 0   # number of Saturation Pixels that are Bright in Green Region

        val_on_pixels_red = 0     # number of Value Pixels that are Bright in Red Region
        val_on_pixels_green = 0   # number of Value Pixels that are Bright in Green Region

        for i in hsv_list_green:
            if i[0] >= self.green_params["hue_low"] and i[0] <= self.green_params["hue_high"]:
                hue_sum_green += i[0]
                length_green += 1
                if i[1] >= self.green_params["sat_low"] and i[1] <= self.green_params["sat_high"]:
                    sat_on_pixels_green += 1
                if i[2] >= self.green_params["val_low"] and i[2] <= self.green_params["val_high"]:
                    val_on_pixels_green += 1

        for j in hsv_list_red:
            if j[0] >= self.red_params["hue_low"] and j[0] <= self.red_params["hue_high"]:
                hue_sum_red += j[0]
                length_red += 1
                if j[1] >= self.red_params["sat_low"] and j[1] <= self.red_params["sat_high"]:
                    sat_on_pixels_red += 1
                if j[2] >= self.red_params["val_low"] and j[2] <= self.red_params["val_high"]:
                    val_on_pixels_red += 1

        #hue_intensity_green = (hue_on_pixels_green/len(hsv_list_green)) * 100
        sat_intensity_green = (sat_on_pixels_green/len(hsv_list_green)) * 100
        val_intensity_green = (val_on_pixels_green/len(hsv_list_green)) * 100

        #hue_intensity_red = (hue_on_pixels_red/len(hsv_list_red)) * 100
        sat_intensity_red = (sat_on_pixels_red/len(hsv_list_red)) * 100
        val_intensity_red = (val_on_pixels_red/len(hsv_list_red)) * 100

        if length_green == 0:
            length_green = 1
        if length_red == 0:
            length_red = 1

        # Mean of Hue
        hsv_mean_green = hue_sum_green / length_green
        hsv_mean_red = hue_sum_red / length_red

        # Detection based on color Threshold 
        # Day mode (different ROI posision and color threshold)
        if (sat_intensity_green > sat_intensity_red) and (val_intensity_green > val_intensity_red):
            if hsv_mean_green >= self.green_params["hue_low"] and hsv_mean_green <= self.green_params["hue_high"]:
                return 'Green'
        elif (sat_intensity_red > sat_intensity_green) and (val_intensity_red > val_intensity_green):
            if (hsv_mean_red >= self.red_params["hue_low"] and hsv_mean_red <= self.red_params["hue_high"]):
                return 'Red'
        else:
            return 'Not Detected'

        return 'Not Detected'

    # Run main
    def run(self):
        if self.target_mode == 'Video':
            # data store
            output_data = []
            # Display window parameters
            coordinates = (1495, 280)
            # undetected frame counter
            counter = 0

            # timer
            start = time.time()

            if self.input.isOpened() == False:
                print('Error openning video file')
            while(self.input.isOpened()):
                ret, frame = self.input.read()
                if ret == True:

                    # Data Storage
                    state_data = {}
                    
                    # Region of Interest
                    img_red = self.region_of_interest(frame, self.red_region)
                    img_green = self.region_of_interest(frame, self.green_region)

                    # Light State
                    state = self.light_state([img_red, img_green])
                    state_data['State'] = state

                    if state == self.state_list[1]:
                        frame = cv2.putText(frame, state, coordinates, self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif state == self.state_list[0]:
                        frame = cv2.putText(frame, state, coordinates, self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    elif state == 'Not Detected':
                        frame = cv2.putText(frame, state, coordinates, self.font, 1, (200, 0, 0), 2, cv2.LINE_AA)
                        cv2.imwrite('./not detected/img' + str(counter) + '.jpg', frame)
                        counter += 1
                    cv2.imshow('Video', frame)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

                    # if state == 'Not Detected':
                    #     cv2.imwrite(self.not_detected_path + '/img' + str(counter) + '.jpg', frame)
                    #     counter += 1

                    output_data.append(state_data)
                else:
                    break
            
            # releasing video memory
            self.input.release()

            print('Done processing video!')
            # State data final dumb into csv file
            df = pd.DataFrame(output_data)
            df.to_csv(self.csv_data_final, index=False, lineterminator='\n')
            print('Done Writing CSV data!')
            end = time.time()
            print('Execution time: ', end - start)
            cv2.destroyAllWindows()

        elif self.target_mode == 'Image':
            # Display window parameters
            coordinates = (1495, 280)

            # Region of Interest
            frame = self.input
            img_red = self.region_of_interest(frame, self.red_region)
            img_green = self.region_of_interest(frame, self.green_region)

            # Image Visualizer
            # self.frame_visualizer([img_green])

            rgb_list_red = self.RGB_list(img_red)
            rgb_list_green = self.RGB_list(img_green)
            
            hsv_list_red = self.HSV_list(rgb_list_red)
            hsv_list_green = self.HSV_list(rgb_list_green)

            # Light State
            state = self.light_state([img_red, img_green])
            print(state)
            cv2.imshow('Video', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    # Argument Parser
    def argument_parser():
        ap = argparse.ArgumentParser()
        # ap.add_argument("--input", type=str, default='./videos/new_video.mp4', help=("path to the input file", "e.g. .mkv .mp4 .jpg .png"))
        ap.add_argument("--input", type=str, default='./images/img6.jpg', help=("path to the input file", "e.g. .mkv .mp4 .jpg .png"))
        ap.add_argument("--region", type=list, default = [[350, 1504, 354, 1508], [358, 1504, 362, 1508], [217, 1464, 220, 1467], [224, 1463, 227, 1466]], help="list of region dimensions eg. [x, y, width, height]")
        ap.add_argument("--night_mode", type=bool, default=False, help='detect in night video or day video')
        ap.add_argument("--output_dir", type=str, default=None, help='output directory path')
        args = ap.parse_args()
        return args
    
    # Main Func
    def main(args):
        ROI(**vars(args))

# Main Method
if '__main__' == __name__:
    input_args = ROI.argument_parser()
    print('--------Carissma--------')
    print('Processing.....')
    ROI.main(input_args)