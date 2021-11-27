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
        for detection in xyxy:
            x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
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
