from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import os
from kalmanfilter import KalmanFilter
from heading_angle import Angle

class Minimap():
    def __init__(self, minimap_type='Terrain', minimap_coords=((1423, 710), (1865, 1030)), trajectory_update_rate=30, trajectory_retain_duration=250):
        self.homography_CameraToMap = np.load('./map_files/homography_CameraToMap.npy')
       
        if minimap_type == 'Terrain':
            self.Minimap = cv2.imread('./map_files/map_satellite_cropped.png')
        elif minimap_type == 'Road':
            self.Minimap = cv2.imread('./map_files/map_cropped.png')
        else:
            print("Wrong Minimap type...defaulting to 'Terrain'")
            self.Minimap = cv2.imread('./map_files/map_satellite_cropped.png')
        
        # Location in the main image to insert minimap
        self.locationMinimap = minimap_coords
        original_width = self.Minimap.shape[1]
        original_height = self.Minimap.shape[0]
 
        # Resizing the minimap accordingly
        resize_width = self.locationMinimap[1][0] - self.locationMinimap[0][0]
        resize_height = self.locationMinimap[1][1] - self.locationMinimap[0][1]

        self.Minimap = cv2.resize(self.Minimap, (resize_width, resize_height))

        self.width_scaling = resize_width/original_width
        self.height_scaling = resize_height/original_height

        self.realtime_trajectory = defaultdict(list)
        self.updateRate = trajectory_update_rate
        self.trajectory_retain_duration = trajectory_retain_duration

    def projection_image_to_map(self, x, y):
        """Converts image coordinates to minimap coordinates using loaded the homography matrix

        Returns:
          (int, int): x, y coordinates with respective to scaled minimap
        """
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.homography_CameraToMap, pt1)
        pt2 = pt2 / pt2[2]
        return (int(pt2[0]*self.width_scaling), int(pt2[1]*self.height_scaling))

    def projection_image_to_map_noScaling(self, x, y):
        """Converts image coordinates to minimap coordinates using loaded the homography matrix

        Returns:
          (int, int): x, y coordinates with respective to scaled minimap
        """
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.homography_CameraToMap, pt1)
        pt2 = pt2 / pt2[2]
        return (int(pt2[0]), int(pt2[1]))
    
    def update_realtime_trajectory(self, current_frameNumber):
        """Responsible for deleting trajectory points for each tracker id after 'self.trajectory_retain_duration' frames

        Returns:
            None
        """
        if self.realtime_trajectory:
            for keys, values in list(self.realtime_trajectory.items()):
                if len(values) == 0:
                    del self.realtime_trajectory[keys]
                elif current_frameNumber - values[0][3] > self.trajectory_retain_duration:
                    del self.realtime_trajectory[keys][0]

 
class Visualizer():
    def __init__(self, minimap, trajectory_mode, trajectory_update_rate, trajectory_retain_duration, save_class_frames):
        self.classID_dict = {
            0: ("Escooter", (0, 90, 255)), 
            1: ("Pedestrians", (255, 90, 0)), 
            2: ("Cyclists", (90, 255, 0)),
            3: ("Motorcycle", (204, 0, 102)),
            4: ("Car", (0, 0, 255)),
            5: ("Truck", (0, 102, 204)),
            6: ("Bus", (0, 255, 255))
        }

        self.textColor = (0, 0, 0)
        
        self.count = 0  # variable to update default_dict after certain number of count
        self.track_count = 0
        self.save_class_frames = save_class_frames
        self.kf = KalmanFilter()
        self.angle = Angle()

        if minimap:
            print(f"[INFO] Minimap is set to {minimap}")
            self.showMinimap = True
            self.Minimap_obj = Minimap(trajectory_update_rate=trajectory_update_rate, trajectory_retain_duration=trajectory_retain_duration)
            self.showTrajectory = trajectory_mode
        else:
            self.showMinimap = False
            self.showTrajectory = False
    
    def draw_realtime_trajectory(self, minimap_img):
        """Displays the recorded trajectory onto the minimap

        Returns:
            None
        """
        if self.Minimap_obj.realtime_trajectory:
            for keys, values in self.Minimap_obj.realtime_trajectory.items():
                for v in values:
                    color = self.classID_dict[v[2]][1]
                    cv2.circle(minimap_img, (int(v[0]),int(v[1])), 1, color, -1, cv2.LINE_AA)
        return minimap_img 

    def drawEmpty(self, frame, frameCount):
        """For images with no detections, displaying minimap and updating trajectory values
        
        Returns:
            frame (image): Image with minimap (if minimap enabled)
        """     
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            if self.showTrajectory:
                minimap_img = self.draw_realtime_trajectory(minimap_img)
                self.Minimap_obj.update_realtime_trajectory(frameCount)
            frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img
            return frame
        else:
            return frame
            

    def drawBBOX(self, xyxy, frame, frameCount):
        """Draws just the BBOX with the class name and confidence score

        Args:
            xyxy (array): output from inference
            frame (image): Image to draw

        Returns:
            frame (image): Image with all the BBOXes
        """
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            if self.showTrajectory:
                minimap_img = self.draw_realtime_trajectory(minimap_img)
                self.Minimap_obj.update_realtime_trajectory(frameCount)
        
        for detection in xyxy:  
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            try:
                conf_score = round(detection[4].item() * 100, 1)
            except AttributeError:
                conf_score = round(detection[4] * 100, 1)

            try:
                classID = int(detection[5].item())
            except AttributeError:
                classID = int(detection[5])
                
            color = self.classID_dict[classID][1]
            
            # Displays the main bbox and add overlay to make bbox transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Finds the space required for text
            textLabel = f'{self.classID_dict[classID][0]} {conf_score}%'
            (w1, h1), _ = cv2.getTextSize(
                textLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Displays BG Box for the text and text itself
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            frame = cv2.putText(
                image, textLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )

            if self.showMinimap:
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                _, max_y = sorted((y1, y2))
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)
                cv2.circle(minimap_img, point_coordinates, 1, color, -1, cv2.LINE_AA)

        if self.showMinimap:      
            frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img

        return frame

    def drawTracker(self, trackers, frame, frameCount):
        """Draws the BBOX along with Tracker ID for each BBOX

        Args:
            trackers (array): SORT Tracker object
            frame (image): Image to draw

        Returns:
            image: Image with tracker id and bbox
        """

        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            if self.showTrajectory:
                minimap_img = self.draw_realtime_trajectory(minimap_img)
                self.Minimap_obj.update_realtime_trajectory(frameCount)
        
        for detection in trackers:
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            conf_score = round(detection[4] * 100, 1)
            classID = int(detection[5])
            tracker_id = int(detection[9])

            color = self.classID_dict[classID][1]
            
            # Displays the main bbox and add overlay to make bbox transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Finds the space required for text
            TrackerLabel = f'Track ID: {tracker_id}'
            (w1, h1), _ = cv2.getTextSize(
                TrackerLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            baseLabel = f'{self.classID_dict[classID][0]} {conf_score}%'
            (w2, h2), _ = cv2.getTextSize(
                baseLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Displays BG Box for the text and text itself
            cv2.rectangle(overlay, (x1, y1 - 40), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w2, y1), color, -1, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            frame = cv2.putText(
                image, TrackerLabel, (x1, y1 - 24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                image, baseLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )

            if self.showMinimap:
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                _, max_y = sorted((y1, y2))
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)

                if self.showTrajectory:
                  if frameCount % self.Minimap_obj.updateRate == 0:
                      self.Minimap_obj.realtime_trajectory[tracker_id].append((point_coordinates[0], point_coordinates[1], classID, frameCount))
            
                cv2.circle(minimap_img, point_coordinates, 1, color, -1, cv2.LINE_AA)
                  
                # Plotting the text
                textSize, _ = cv2.getTextSize(str(tracker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                rectangle_start_coord = (point_coordinates[0] + 3, point_coordinates[1] - textSize[1] - 5)
                rectangle_end_coord = (point_coordinates[0] + textSize[0] + 3, point_coordinates[1])
                
                cv2.rectangle(minimap_img, rectangle_start_coord, rectangle_end_coord, color, -1)
                cv2.putText(minimap_img, str(tracker_id), tuple((point_coordinates[0] + 3, point_coordinates[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA)
        
        if self.showMinimap:
            frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img

        return frame

    def drawAll(self, trackers, frame, frameCount, output):
        """Draws the BBOX along with Tracker ID and speed for every detection

        Args:
            trackers (array): SORT Tracker object (with speed(kmh) as the last element)
            frame (image): Image to draw

        Returns:
            image: Image with tracker id, speed(kmh) and bbox
        """

        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            if self.showTrajectory:
                minimap_img = self.draw_realtime_trajectory(minimap_img)
                self.Minimap_obj.update_realtime_trajectory(frameCount)
        self.count +=1
        output_path = output

        for detection in trackers:
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            conf_score = round(detection[4] * 100, 1)
            classID = int(detection[5])
            tracker_id = int(detection[9])
            speed = detection[-4]
            color = self.classID_dict[classID][1]

            # variables for heading arrow
            cx1, cy1 = int(detection[-3]), int(detection[-2])   # previous frame points
            track_pts = detection[-1]
            
            # Displays the main bbox and add overlay to make bbox transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Finds the space required for text
            TrackerLabel = f'Track ID: {tracker_id}'
            (w1, h1), _ = cv2.getTextSize(
                TrackerLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            speedLabel = f'Speed: {speed}km/h'
            (w2, h2), _ = cv2.getTextSize(
                speedLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            baseLabel = f'{self.classID_dict[classID][0]} {conf_score}%'
            (w3, h3), _ = cv2.getTextSize(
                baseLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Displays BG Box for the text and text itself
            cv2.rectangle(overlay, (x1, y1 - 60), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1 - 40), (x1 + w2, y1), color, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w3, y1), color, -1, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            frame = cv2.putText(
                image, TrackerLabel, (x1, y1 - 43), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                image, speedLabel, (x1, y1 - 24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                image, baseLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )

            # Save frames of required Class Instances in 30 frames consicutive for eeach tracker_id of that class. 
            if classID == self.save_class_frames and self.track_count<2:
                output_path_dir = os.path.join(output_path, "Save-frames")
                cv2.imwrite(fr"{output_path_dir}\{self.classID_dict[classID][0]}-{tracker_id}_count-{self.track_count}.jpg", frame)
                self.track_count += 1
                
            # Use kalman_filter to predict next point and draw heading arrow in that direction
            if type(track_pts)==defaultdict:
                for pt in track_pts[tracker_id]:
                    predicted = self.kf.predict(pt[0], pt[1])
                pred = self.kf.predict(predicted[0], predicted[1])
                pred2 = self.kf.predict(pred[0], pred[1])
                cv2.arrowedLine(frame, (cx1,cy1), (int(pred2[0]),int(pred2[1])), (255,0,0),1)
                points = [[cx1, cy1], [int(pred2[0]), int(pred2[1])], [x2, cy1]]
                if speed!= 0:
                    angle = self.angle.findangle(points=points)
                    cv2.arrowedLine(frame, (cx1,cy1), (x2, cy1),  (0,0,255),1)
                    AngleLabel = f'Angle: {angle}'
                    (w4, h4), _ = cv2.getTextSize(
                        AngleLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    cv2.rectangle(frame, (x2, y1 + 20), (x2 + w4, y1), color, -1, cv2.LINE_AA)
                    frame = cv2.putText(
                    image, AngleLabel, (x2, y1 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
                    )     

            if self.showMinimap:
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                if classID in (0,1,2):       
                    _, max_y = sorted((y1, y2))
                elif classID in (3,4,5,6):
                    max_y = int((y1+y2)/2)   # Center of bbox for classes other than Escooter, Cyclist, and Pedestrian
                
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)

                if self.showTrajectory:
                  if frameCount % self.Minimap_obj.updateRate == 0:
                      self.Minimap_obj.realtime_trajectory[tracker_id].append((point_coordinates[0], point_coordinates[1], classID, frameCount))
            
                cv2.circle(minimap_img, point_coordinates, 1, color, -1, cv2.LINE_AA)
                  
                # Plotting the text
                textSize, _ = cv2.getTextSize(str(tracker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                rectangle_start_coord = (point_coordinates[0] + 3, point_coordinates[1] - textSize[1] - 5)
                rectangle_end_coord = (point_coordinates[0] + textSize[0] + 3, point_coordinates[1])
                
                cv2.rectangle(minimap_img, rectangle_start_coord, rectangle_end_coord, color, -1)
                cv2.putText(minimap_img, str(tracker_id), tuple((point_coordinates[0] + 3, point_coordinates[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA)
        
        if self.showMinimap:
            frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img

        return frame
