from Inference_Class import Inference
from OCR import OCR
import os

video_sample_path = os.path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\samples')
model_weights = os.path.abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\2_7250\model_final.pth")
test_dataset_path = os.path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')


def process():
    
    for videos in os.listdir(video_sample_path):    
        if videos[-4:] in ['.mkv', '.mp4', '.avi']:
            print(videos)
            video_path = video_sample_path + f'//{videos}'
            inference = Inference(model_weights, test_dataset_path, video_path, mode='Video')
            inference.
