# importing the module
import cv2
   
# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x/2) + ',' +
                    str(y/2), (x+2,y+2), font,
                    1, (255, 0, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow('image', img)
  
# driver function
if __name__=="__main__":
  
    # reading the image
    img = cv2.imread('/home/mobilitylabextreme002/Desktop/Traffic_Camera_Tracking/map_files/map_satellite_cropped.png', 1)
    print(img.shape)
    img = cv2.resize(img, [img.shape[0]*2, img.shape[1]*2])
    # displaying the image
    cv2.imshow('image', img)
  
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
  
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    cv2.imwrite("outputs/lat_long_minimap_output.png", img)
    # close the window
    cv2.destroyAllWindows()