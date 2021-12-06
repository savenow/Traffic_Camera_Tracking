import cv2
import numpy as np
from collections import defaultdict

class Minimap():
    def __init__(self, minimap_type='Terrain', minimap_coords=((1423, 710), (1865, 1030)), trajec_coords = ((732, 234), (1181, 544))):
        self.homography_CameraToMap = np.load('./map_files/homography_CameraToMap.npy')
       
        if minimap_type == 'Terrain':
            self.Minimap = cv2.imread('./map_files/map_satellite_cropped.png')
        elif minimap_type == 'Road':
            self.Minimap = cv2.imread('./map_files/map_cropped.png')
        else:
            print("Wrong Minimap type...defaulting to 'Terrain'")
            self.Minimap = cv2.imread('./map_files/map_satellite_cropped.png')
        
        self.trajecmap = cv2.imread('./map_files/map_satellite_cropped.png')

        # Location in the main image to insert minimap
        self.locationMinimap = minimap_coords
        
        original_width = self.Minimap.shape[1]
        original_height = self.Minimap.shape[0]

        # Location in the main image to insert trajectory_map
        self.locationTrajecmap = trajec_coords
        o_w = self.trajecmap.shape[1]
        o_h = self.trajecmap.shape[0]
        # Resize trajectory_map
        w = self.locationTrajecmap[1][0] - self.locationTrajecmap[0][0]
        h = self.locationTrajecmap[1][1] - self.locationTrajecmap[0][1]
        self.trajecmap = cv2.resize(self.trajecmap, (w, h))
        self.scaled_w = w/o_w
        self.scaled_h = h/o_h
 
        # Resizing the minimap accordingly
        resize_width = self.locationMinimap[1][0] - self.locationMinimap[0][0]
        resize_height = self.locationMinimap[1][1] - self.locationMinimap[0][1]

        self.Minimap = cv2.resize(self.Minimap, (resize_width, resize_height))

        self.width_scaling = resize_width/original_width
        self.height_scaling = resize_height/original_height

    def projection_image_to_map(self, x, y):
        #x, y = image_coordinates
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.homography_CameraToMap, pt1)
        pt2 = pt2 / pt2[2]
        return (int(pt2[0]*self.width_scaling), int(pt2[1]*self.height_scaling))

    def trajectory_img_to_map(self,x,y):
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.homography_CameraToMap, pt1)
        pt2 = pt2 / pt2[2]
        return (int(pt2[0]), int(pt2[1]))

    def map_trajectory_capture(self, x, y):
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.homography_CameraToMap, pt1)
        pt2 = pt2 / pt2[2]
        return (int(pt2[0]*self.scaled_w), int(pt2[1]*self.scaled_h))


class Visualizer():
    def __init__(self, minimap=False, trajectory_mode = False):
        self.classID_dict = {
            0: ("Escooter", (0, 90, 255)), 
            1: ("Pedestrians", (255, 90, 0)), 
            2: ("Cyclists", (90, 255, 0))
        }
        self.textColor = (255, 255, 255)
        
        self.draw_trajectory_on_map = defaultdict(list)
        self.realtime_trajectory = defaultdict(list)
        self.draw_trajectory_on_img = defaultdict(list)

        if minimap:
            self.showMinimap = True
            self.Minimap_obj = Minimap()
            
    def draw_realtime_trajectory(self, realtime_trajectory, trajec_img, frame):
        if realtime_trajectory != None:
            for key, value in realtime_trajectory.items():
                if key == 0:
                    color = self.classID_dict[key][1]
                    for v in value:
                        cv2.circle(trajec_img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                        frame[self.Minimap_obj.locationTrajecmap[0][1]:self.Minimap_obj.locationTrajecmap[1][1], self.Minimap_obj.locationTrajecmap[0][0]:self.Minimap_obj.locationTrajecmap[1][0]] = trajec_img
                if key == 1:
                    color = self.classID_dict[key][1]
                    for v in value:
                        cv2.circle(trajec_img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                        frame[self.Minimap_obj.locationTrajecmap[0][1]:self.Minimap_obj.locationTrajecmap[1][1], self.Minimap_obj.locationTrajecmap[0][0]:self.Minimap_obj.locationTrajecmap[1][0]] = trajec_img
                if key == 2:
                    color = self.classID_dict[key][1]
                    for v in value:
                        cv2.circle(trajec_img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                        frame[self.Minimap_obj.locationTrajecmap[0][1]:self.Minimap_obj.locationTrajecmap[1][1], self.Minimap_obj.locationTrajecmap[0][0]:self.Minimap_obj.locationTrajecmap[1][0]] = trajec_img
        
            return frame
            
    def drawBBOX(self, xyxy, frame):
        """Draws just the BBOX with the class name and confidence score

        Args:
            xyxy (array): output from inference
            frame (image): Image to draw

        Returns:
            frame (image): Image with all the BBOXes
        """
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            trajec_img = self.Minimap_obj.trajecmap.copy()

        for detection in xyxy:  
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            conf_score = round(detection[4].item() * 100, 1)
            classID = int(detection[5].item())

            color = self.classID_dict[classID][1]
            
            # Displays the main bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Finds the space required for text
            textLabel = f'{self.classID_dict[classID][0]} {conf_score}%'
            (w1, h1), _ = cv2.getTextSize(
                textLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Displays BG Box for the text and text itself
            frame = cv2.rectangle(frame, (x1, y1 - 20), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            frame = cv2.putText(
                frame, textLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )

            if self.showMinimap:
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                _, max_y = sorted((y1, y2))
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)
                point_trajecmap = self.Minimap_obj.map_trajectory_capture((x1+x2)/2, max_y)
                pt_trajectory = self.Minimap_obj.trajectory_img_to_map((x1+x2)/2, max_y)

                self.draw_trajectory_on_map[classID].append(tuple(pt_trajectory))
                self.draw_trajectory_on_img[classID].append((int((x1+x2)/2), int(max_y)))
                self.realtime_trajectory[classID].append(tuple(point_trajecmap))

                cv2.circle(minimap_img, tuple(point_coordinates), 3, color, -1, cv2.LINE_AA)
                frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img
                
                frame = self.draw_realtime_trajectory(self.realtime_trajectory, trajec_img, frame)

        return frame

    def drawTracker(self, trackers, frame):
        """Draws the BBOX along with Tracker ID for each BBOX

        Args:
            trackers (array): SORT Tracker object
            frame (image): Image to draw

        Returns:
            image: Image with tracker id and bbox
        """
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            trajec_img = self.Minimap_obj.trajecmap.copy()
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
            
            # Displays the main bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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
            frame = cv2.rectangle(frame, (x1, y1 - 40), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (x1, y1 - 20), (x1 + w2, y1), color, -1, cv2.LINE_AA)
            
            frame = cv2.putText(
                frame, TrackerLabel, (x1, y1 - 24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                frame, baseLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )

            if self.showMinimap:
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                _, max_y = sorted((y1, y2))
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)
                point_trajecmap = self.Minimap_obj.map_trajectory_capture((x1+x2)/2, max_y)
                pt_trajectory = self.Minimap_obj.trajectory_img_to_map((x1+x2)/2, max_y)

                self.draw_trajectory_on_map[classID].append(tuple(pt_trajectory))
                self.draw_trajectory_on_img[classID].append((int((x1+x2)/2), int(max_y)))
                self.realtime_trajectory[classID].append(tuple(point_trajecmap))

                cv2.circle(minimap_img, tuple(point_coordinates), 3, color, -1, cv2.LINE_AA)
                frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img
                
                frame = self.draw_realtime_trajectory(self.realtime_trajectory, trajec_img, frame)
                
                # Plotting the text
                textSize, _ = cv2.getTextSize(str(tracker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                rectangle_start_coord = (point_coordinates[0] + 3, point_coordinates[1] - textSize[1] - 5)
                rectangle_end_coord = (point_coordinates[0] + textSize[0] + 3, point_coordinates[1])
                
                cv2.rectangle(minimap_img, rectangle_start_coord, rectangle_end_coord, color, -1)
                cv2.putText(minimap_img, str(tracker_id), tuple((point_coordinates[0] + 3, point_coordinates[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA)
        
        return frame

    def drawAll(self, trackers, frame):
        """Draws the BBOX along with Tracker ID and speed for every detection

        Args:
            trackers (array): SORT Tracker object (with speed(kmh) as the last element)
            frame (image): Image to draw

        Returns:
            image: Image with tracker id, speed(kmh) and bbox
        """
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            trajec_img = self.Minimap_obj.trajecmap.copy()
    
        for detection in trackers:
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            conf_score = round(detection[4] * 100, 1)
            classID = int(detection[5])
            tracker_id = int(detection[9])
            speed = detection[-1]
            
            color = self.classID_dict[classID][1] 
            
            # Displays the main bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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
            frame = cv2.rectangle(frame, (x1, y1 - 60), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (x1, y1 - 40), (x1 + w2, y1), color, -1, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (x1, y1 - 20), (x1 + w3, y1), color, -1, cv2.LINE_AA)
            
            frame = cv2.putText(
                frame, TrackerLabel, (x1, y1 - 43), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                frame, speedLabel, (x1, y1 - 24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                frame, baseLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )

            if self.showMinimap:      
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                _, max_y = sorted((y1, y2))
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)
                point_trajecmap = self.Minimap_obj.map_trajectory_capture((x1+x2)/2, max_y)
                pt_trajectory = self.Minimap_obj.trajectory_img_to_map((x1+x2)/2, max_y)

                self.draw_trajectory_on_map[classID].append(tuple(pt_trajectory))
                self.draw_trajectory_on_img[classID].append((int((x1+x2)/2), int(max_y)))
                self.realtime_trajectory[classID].append(tuple(point_trajecmap))

                cv2.circle(minimap_img, tuple(point_coordinates), 3, color, -1, cv2.LINE_AA)
                frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img
                
                frame = self.draw_realtime_trajectory(self.realtime_trajectory, trajec_img, frame)

                # Plotting the text
                textSize, _ = cv2.getTextSize(str(tracker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                rectangle_start_coord = (point_coordinates[0] + 3, point_coordinates[1] - textSize[1] - 5)
                rectangle_end_coord = (point_coordinates[0] + textSize[0] + 3, point_coordinates[1])
                
                cv2.rectangle(minimap_img, rectangle_start_coord, rectangle_end_coord, color, -1)
                cv2.putText(minimap_img, str(tracker_id), tuple((point_coordinates[0] + 3, point_coordinates[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA)
                
        return frame

    def draw_All_trejectory(self, draw_trajectory_on_map, draw_trajectory_on_img, trajectory_mode, result_type= 'On_map'):
        if trajectory_mode == True:
            if result_type == "On_map":
                img = cv2.imread('./map_files/map_satellite_cropped.png')
                for key, value in draw_trajectory_on_map.items():
                    if key == 0:
                        color = self.classID_dict[key][1]
                        for v in value:
                            img = cv2.circle(img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                    elif key == 1:
                        color = self.classID_dict[key][1]
                        for v in value:
                            img = cv2.circle(img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                    elif key == 2:
                        color = self.classID_dict[key][1]
                        for v in value:
                            img = cv2.circle(img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                return img

            elif result_type == "On_image":
                img = cv2.imread('./map_files/image.jpeg')
                for key, value in draw_trajectory_on_img.items():
                    if key == 0:
                        color = self.classID_dict[key][1]
                        for v in value:
                            img = cv2.circle(img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                    elif key == 1:
                        color = self.classID_dict[key][1]
                        for v in value:
                            img = cv2.circle(img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                    elif key == 2:
                        color = self.classID_dict[key][1]
                        for v in value:
                            img = cv2.circle(img, (v[0],v[1]), 1, color, -1, cv2.LINE_AA)
                return img

        else:
            None
    
    