import cv2

class Visualizer():
    def __init__(self, minimap=False, minimap_type='Road', minimap_img_location=((1423, 710), (1865, 1030))):
        self.classID_dict = {
            0: ("Escooter", (0, 90, 255)), 
            1: ("Pedestrians", (255, 90, 0)), 
            2: ("Cyclists", (90, 255, 0))
        }
        self.textColor = (255, 255, 255)
        
        if minimap:
            self.showMinimap = True
            if minimap_type == 'Terrain':
                self.Minimap = cv2.imread('./map_files/map_satellite_cropped.png')
            elif minimap_type == 'Road':
                self.Minimap = cv2.imread('./map_files/map_cropped.png')
            else:
                print("Wrong Minimap type...defaulting to 'Terrain'")
                self.Minimap = cv2.imread('./map_files/map_satellite_cropped.png')

            # Location in the main image to insert minimap
            self.locationMinimap = minimap_img_location
            
            # Resizing the minimap accordingly
            resize_width = self.locationMinimap[1][0] -self.locationMinimap[0][0]
            resize_height = self.locationMinimap[1][1] - self.locationMinimap[0][1]

            self.Minimap = cv2.resize(self.Minimap, (resize_width, resize_height))

    def drawBBOX(self, xyxy, frame):
        """Draws just the BBOX with the class name and confidence score

        Args:
            xyxy (array): output from inference
            frame (image): Image to draw

        Returns:
            frame (image): Image with all the BBOXes
        """
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
        
        return frame

    def drawTracker(self, trackers, frame):
        """Draws the BBOX along with Tracker ID for each BBOX

        Args:
            trackers (array): SORT Tracker object
            frame (image): Image to draw

        Returns:
            image: Image with tracker id and bbox
        """
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
            minimap_img = self.Minimap.copy()
            minimap_points = []
    
        for detection in trackers:
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            conf_score = round(detection[4] * 100, 1)
            classID = int(detection[5])
            tracker_id = int(detection[9])
            speed = detection[-3]
            
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
                minimap_points.append((int(detection[-2]), int(detection[-1]), color))
                #print(minimap_point)

        if self.showMinimap:
          for point in minimap_points:
            cv2.circle(minimap_img, tuple(point[0:2]), 2, point[2], -1, cv2.LINE_AA)

          frame[self.locationMinimap[0][1]:self.locationMinimap[1][1], self.locationMinimap[0][0]:self.locationMinimap[1][0]] = minimap_img      
        
        return frame