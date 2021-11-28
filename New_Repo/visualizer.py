import cv2

class Visualizer():
    def __init__(self):
        self.classID_dict = {
            0: ("Escooter", (0, 90, 255)), 
            1: ("Pedestrians", (255, 90, 0)), 
            2: ("Cyclists", (90, 255, 0))
        }
        self.textColor = (255, 255, 255)

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
        
        return frame