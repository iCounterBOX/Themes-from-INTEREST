"""
Extract CAR Windshield from our CAR PICS
1. Its showing just ONE of our pics - we can draw the ROI-SCTION
2. From This ROI we extract the WindChield-Picture
    

cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
x1,y1 ------
|          |
|          |
|          |
--------x2,y2


"""
import os
import cv2
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

camera_height = 500

#rect

x1 = 65
y1 = 128
x2 = 753
y2 = 412

IMG_SIZE = 800

directory = './detectedImages/'

_cCnt = 0


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
        cv2.imshow("carOccupancy", myPic)
        
        global x1,y1
        x1,y1 = ref_point[0]
        global x2,y2
        x2,y2 = ref_point[1]
        
        
        #https://stackoverflow.com/questions/23720875/how-to-draw-a-rectangle-around-a-region-of-interest-in-python
        print("x1,y1 = %s" %str( ref_point[0])  + " x2,y2 = %s" %str( ref_point[1]))






# We start to load just ONE picture...and in a loop we just set our ROI-Frame

cv2.namedWindow("carOccupancy")
cv2.setMouseCallback("carOccupancy", shape_selection)
            
picPath = os.path.join(directory, 'car162.jpg')
myPic = cv2.imread(picPath,0)            
myPic = cv2.resize(myPic, (IMG_SIZE, IMG_SIZE) )
clone =  myPic.copy()
cv2.imshow("carOccupancy", myPic) 

while(True):
            key = cv2.waitKey(1)
                    
            # press 'r' to reset the window
            if key == ord("r"):               
               myPic = clone.copy()
               cv2.imshow("carOccupancy", myPic) 
                
            # quit camera if 'q' key is pressed
            if key & 0xFF == ord("q"):
                break
cv2.destroyAllWindows()    


_WriteSinglePic2File = True

for img in os.listdir(directory):   
    if img.endswith(".jpg"):         
            
            
            picPath = os.path.join(directory, img)
            myPic = cv2.imread(picPath,0)            
            myPic = cv2.resize(myPic, (IMG_SIZE, IMG_SIZE) )
            cv2.imshow("carOccupancy", myPic)
           

            
            cv2.rectangle( myPic, (x1, y1), (x2, y2), (240, 100, 0), 2)
            # get ROI
            roi = myPic[y1:y2, x1:x2]    
            cv2.imshow('ROI',roi)
            if _WriteSinglePic2File:
                roi = cv2.resize(roi, (400, 300))    
                _cCnt += 1
                newImgPath = './detectedImages/winshields/carFW' + str(_cCnt) + '.jpg' 
                #print('new filename: ' +  newImgPath)
                cv2.imwrite(newImgPath, roi)     #  <<<<---- WRITE THE SINGLE PIC TO folder
                       
            # show the frame
            cv2.imshow("carOccupancy", myPic)
                
            key = cv2.waitKey(1)                   
         
                
            # quit camera if 'q' key is pressed
            if key & 0xFF == ord("q"):
                break
            continue
    else:
        continue
    
    break

cv2.destroyAllWindows()




