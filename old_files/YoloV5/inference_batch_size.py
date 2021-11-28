import torch
import time
import cv2
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from statistics import mean


class TqdmExtraFormat(tqdm):
    """Provides a `total_time` format parameter"""
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time) + " in total")
        return d


def video_inference(input_path, output_path, model_weights, bs):
    video_capture = cv2.VideoCapture(input_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    print(1/fps)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    codec = 'mp4v'

    video_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height),)
    model = torch.hub.load(
        '/home/students-fkk/yolov5', 
        'custom', 
        path=model_weights, 
        source='local'
    ) 
    model.eval()
    model.cuda()
    model.half()

    device = torch.device('cuda:0')
    cudnn.benchmark = True
    
    batch_size = bs
    batch_count = 0
    tqdm_count = 1
    imgs = []
    fps_per_batch_total = []
    fps_total = []
    frame_count = 0

    pbar = TqdmExtraFormat(total = total_frames, desc='Inference Progress: ')
    while video_capture.isOpened():
        _, frame = video_capture.read()    
        if _:    
            frame_count += 1     
            pbar.update(tqdm_count)
            
            if batch_count != batch_size:
              imgs.append(frame)
              batch_count += 1
            
            else:       
              if batch_size == 1:
                imgs.append(frame)
                 
              t1 = time.time()
              results = model(imgs, size=1280)
              t2 = time.time()
              fps_per_batch = round((1/(t2 - t1)), 3)
              fps_per_batch_total.append(fps_per_batch)
              fps_total.append(fps_per_batch * batch_size)
              
              del imgs
              imgs = []
              batch_count = 1
              for image in results.render():
                video_output.write(image)
                time.sleep(1/fps)
        else:
            if imgs:
              results = model(imgs, size=1408)
              
              for image in results.render():
                pbar.update(tqdm_count)
                video_output.write(image)
                time.sleep(1/fps)
            
            break
    print(f'Batch size: {batch_size}, FPS/batch: {round(mean(fps_per_batch_total), 2)}, final fps: {round(mean(fps_total), 2)}')
    video_capture.release()
    video_output.release()

# Model
model_weight = 'tl_89k_bs24_im1408_e50.pt'

input_directory = '/media/mydisk/videos/samples/08-06-2021_16-40.mkv'
output_directory = '/home/students-fkk/Videos/output_e50/08-06-2021_16-40_1408.mkv'


for batch in [128]:
  video_inference(input_directory, output_directory, model_weight, batch)