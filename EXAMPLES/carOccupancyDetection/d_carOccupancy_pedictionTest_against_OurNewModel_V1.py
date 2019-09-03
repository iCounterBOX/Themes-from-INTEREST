"""
Prediction of a single Picture for finetuning

- rectangle around object
- percentage of the prediction

cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
x1,y1 ------
|          |
|          |
|          |
--------x2,y2


"""

import cv2
import numpy as np
from keras.models import load_model

width = 96
height = 96

class_names = ['ONE', 'TWO']
model = load_model('.\H5Model\car_occupancy_detection_front.h5')
model.summary()


# now let's initialize the list of reference point
ref_point = []

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        global frame
        cv2.rectangle(frame, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("carOccupancyXX", frame)
        
        global x1,y1
        x1,y1 = ref_point[0]
        global x2,y2
        x2,y2 = ref_point[1]
        
        
        #https://stackoverflow.com/questions/23720875/how-to-draw-a-rectangle-around-a-region-of-interest-in-python
        print("x1,y1 = %s" %str( ref_point[0])  + " x2,y2 = %s" %str( ref_point[1]))


# get the reference to the webcam
# Define the video stream
cap = cv2.VideoCapture('./video/carsInFront.mp4')
camera_height = 500


#rect

x1 = 392
y1 = 159
x2 = 663
y2 = 423

while(True):
    # read a new frame
    ret, frame = cap.read()
   
    cv2.namedWindow("carOccupancy")
    cv2.setMouseCallback("carOccupancy", shape_selection)

    
    if ret:    
                     # flip the frameq
            #frame = cv2.flip(frame, 1)
        
            # rescaling camera output
            aspect = frame.shape[1] / float(frame.shape[0])
            res = int(aspect * camera_height) # landscape orientation - wide image
            frame = cv2.resize(frame, (res, camera_height))
            
            clone = frame.copy()
        
            # add rectangle
            #cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 100, 0), 2)
        
            # get ROI
            roi = frame[y1+2:y2-2, x1+2:x2-2]        # roi = frame[75+2:425-2, 300+2:650-2]
            
            # parse BRG to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
            # resize
            roi = cv2.resize(roi, (width, height))
            #cv2.imshow("ROI - for the Prediction", cv2.resize(roi, (320, 240)))
            
            # predict!
            roi_X = np.expand_dims(roi, axis=0)
        
            predictions = model.predict_proba(roi_X)    #model.predict(roi_X)
            type_1_pred, type_2_pred = predictions[0]
            
            #print("new prediction: " + predictions)
        
            # add text
            type_1_text = '{}: {}%'.format(class_names[0], int(type_1_pred*100))
            cv2.putText(frame, type_1_text, (70, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            if type_1_pred == 1.0:
                cv2.imshow("ONE PERSON", cv2.resize(roi, (320, 240)))
            else:
             img = np.full((100, 100, 3), 127, np.uint8) 
             cv2.imshow('ONE PERSON', img)
                
          
        
            # add text
            tipe_2_text = '{}: {}%'.format(class_names[1], int(type_2_pred*100))
            cv2.putText(frame, tipe_2_text, (70, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            if type_2_pred == 1:
                cv2.imshow("TWO PERSON", cv2.resize(roi, (320, 240)))
            else:
             img = np.full((100, 100, 3), 127, np.uint8) 
             cv2.imshow('TWO PERSON', img)
           
            # show the frame
            cv2.imshow("carOccupancy", frame)
        
            key = cv2.waitKey(1)
            
             # press 'r' to reset the window
            if key == ord("r"):
                frame = clone.copy()
        
            # quit camera if 'q' key is pressed
            if key & 0xFF == ord("q"):
                break

    else:
         break
    
    


cap.release()
cv2.destroyAllWindows()


