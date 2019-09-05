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
import platform
import cv2
import numpy as np
import keras
from keras.models import load_model

width = 96
height = 96

class_names = ['ONE', 'TWO']
model = load_model('.\H5Model\car_occupancy_detection_front.h5')
model.summary()

print("V E R S I O - I N F O:")
print("OpenCV Version: {}".format(cv2.__version__))
print("Python Version: " + platform.python_version())
print("Numpy Version: " +  np.__version__)
print("Keras Version: " +  keras.__version__)


camera_height = 500

IMG_SIZE = 800
#rect

x1 = 524
y1 = 376
x2 = 799
y2 = 469
 

# GET the perfect ROI - Let the Video LOOP        
# https://stackoverflow.com/questions/17158602/playback-loop-option-in-opencv-videos
        
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
        global myPic
        cv2.rectangle(myPic, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("VIDEO-LOOP", myPic)
        
        global x1,y1
        x1,y1 = ref_point[0]
        global x2,y2
        x2,y2 = ref_point[1]       
        
        #https://stackoverflow.com/questions/23720875/how-to-draw-a-rectangle-around-a-region-of-interest-in-python
        print("x1,y1 = %s" %str( ref_point[0])  + " x2,y2 = %s" %str( ref_point[1]))

# We start to load just ONE picture...and in a loop we just set our ROI-Frame

cv2.namedWindow("VIDEO-LOOP")
cv2.setMouseCallback("VIDEO-LOOP", shape_selection)
        
cap = cv2.VideoCapture('./video/carsInFront1.mp4')

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame count of our video: " +  str(frameCount) )   

frame_counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_counter += 1
    #If the last frame is reached, reset the capture and the frame_counter     
    
    if frame_counter == frameCount-2:
        frame_counter = 0 #Or whatever as long as it is the same as next line
        # set video mack to pos 1 
        cap.set(int(cv2.CAP_PROP_POS_FRAMES), 1)
    # Our operations on the frame come here - show current ROI-Rectangle:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (26, 255, 26), 3)       
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    myPic = gray
    clone =  myPic.copy()
    
    # Display the resulting frame
    cv2.imshow('VIDEO-LOOP',gray)
    if cv2.waitKey(100) & 0xFF == ord('x'):
        break
# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()


# Define the video stream

#cap = cv2.VideoCapture('./video/carsInFront1.mp4')



while(True):
    # read a new frame
    ret, frame = cap.read()
   
        
    if ret:    
            # flip the frameq
            #frame = cv2.flip(frame, 1)
            key = cv2.waitKey(50)
        
            # rescaling camera output
            aspect = frame.shape[1] / float(frame.shape[0])
            res = int(aspect * camera_height) # landscape orientation - wide image
            #frame = cv2.resize(frame, (res, camera_height))           
            
        
            # add rectangle
            #cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 100, 0), 2)
        
            # get ROI
            roi = frame[y1+2:y2-2, x1+2:x2-2]        # roi = frame[75+2:425-2, 300+2:650-2]           
            
            # parse BRG to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            #cv2.imshow("ROI - for the Prediction", roi)
        
            # resize
            roix = cv2.resize(roi, (width, height))
            #cv2.imshow("ROI - for the Prediction", cv2.resize(roi, (320, 240)))
            
            # predict!
            roi_X = np.expand_dims(roix, axis=0)
        
            predictions = model.predict_proba(roi_X)    #model.predict(roi_X)
            type_1_pred, type_2_pred = predictions[0]
            print("class: " +  class_names[0] + " prediction: " + str(int(type_1_pred*100)) + " type_1_pred= " + str(type_1_pred) ) 
            print("class: " +  class_names[1] + " prediction: " + str(int(type_2_pred*100)) + " type_2_pred= " + str(type_2_pred) )
            
            #print("new prediction: " + predictions)
        
            # add text
            type_1_text = '{}: {}%'.format(class_names[0], int(type_1_pred*100))
            cv2.putText(frame, type_1_text, (70, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
           
            if type_1_pred == 1.0:
                cv2.imshow("ONE PERSON", roi)
                cv2.destroyWindow('TWO PERSON') 
                         
          
        
            # add text
            tipe_2_text = '{}: {}%'.format(class_names[1], int(type_2_pred*100))
            cv2.putText(frame, tipe_2_text, (70, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
           
            if type_2_pred == 1.0:
                cv2.imshow("TWO PERSON", roi)
                cv2.destroyWindow('ONE PERSON')
                         
           
            # show the frame
            cv2.imshow("carOccupancy", frame)  
        
            # quit camera if 'q' key is pressed
            if key & 0xFF == ord("q"):
                break

    else:
         break
    
    


cap.release()
cv2.destroyAllWindows()


