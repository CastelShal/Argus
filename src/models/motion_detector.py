import cv2
import numpy as np
# import logging

# logger = logging.getLogger(__name__)

class MotionDetector(object):
    def __init__(self):
        self.history = 0 # Keeps track of the number of frames that have been processed
        self.currentFrame = None  
        self.previousFrame = None
        self.meanFrame = None # Frame averaging
        self.person = False
        self.peopleRects = [] # Holds all regions of interest that may contain a person

    def reset_background_model(self):
        self.history = 0

    def detect_movement(self,frame, get_rects):
        # Calculate mean standard deviation then determine if motion has actually accurred
        height, width, _ = frame.shape
        occupied = False
        kernel = np.ones((5,5),np.uint8)

        # Resize the frame, convert it to grayscale, filter and blur it
        # logger.debug('////////////////////// filtering 1 //////////////////////')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # logger.debug('////////////////////// filtering 1.5 //////////////////////')
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray,9)  # Filters out noise
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # logger.debug('////////////////////// filtering 2 //////////////////////')
        
        # Initialise and build background model useing frame averaging
        if self.history <= 3: # Let the camera warm up
            self.currentFrame = gray
            self.history +=1
            if get_rects == True: # Return peoplerects without frame
                return occupied,  self.peopleRects 
            else:
                return occupied,  frame
        
        elif self.history == 4:
            self.previousFrame = self.currentFrame
            self.currentFrame = gray      
            self.meanFrame = cv2.addWeighted(self.previousFrame,0.5,self.currentFrame,0.5,0)
            self.history +=1
            if get_rects == True: # Return peoplerects without frame
                return occupied,  self.peopleRects 
            else:
                return occupied,  frame
        
        elif self.history == 5:
            self.previousFrame = self.meanFrame
            self.currentFrame = gray
            self.meanFrame = cv2.addWeighted(self.previousFrame,0.5,self.currentFrame,0.5,0)
            # cv2.imwrite("avegrayfiltered.jpg", self.meanFrame)
            self.history +=1
            if get_rects == True: # Return peoplerects without frame
                return occupied,  self.peopleRects 
            else:
                return occupied,  frame
        
        elif self.history > 4000 and len(self.peopleRects) == 0: # Recalculate background model every 4000 frames only if there are no people in frame 
            self.previousFrame = self.currentFrame
            self.currentFrame = gray
            self.history = 0

        # logger.debug('////////////////////// averaging complete //////////////////////')
        # Compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(self.meanFrame , gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # Removes small holes i.e noise
        thresh = cv2.dilate(thresh, kernel, iterations=3) # Increases white region by saturating blobs
        # cv2.imwrite("motion.jpg", thresh)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # logger.debug('////////////////////// filtering & thresholding //////////////////////')
        self.peopleRects = []
        # Loop through all contours
        for c in cnts:
            # If the contour is too small or too big, ignore it
            if cv2.contourArea(c) < 2000 or cv2.contourArea(c) > 90000:
                if cv2.contourArea(c) > 100000: # If it is ridiculously big reset background model it is likely that something is wrong
                    self.history = 0
                    break
                continue     
            (x, y, w, h) = cv2.boundingRect(c)  # Compute the bounding box for the contour
            
            # If the bounding box is equal to the width (made smaller never really covers whole width) 
            # or height of the frame it is likely that something is wrong - reset model
            if h == height or w >= width/1.5: 
                    self.history = 0
                    break

            if (h) > (w):  
                occupied = True
                if (h) > (1.5*w): # Most likely a person, this can be made strictor (average human ratio 5.9/1.6 = h/w = 3.6875) 
                    self.person = True
                self.peopleRects.append(cv2.boundingRect(c))
        # logger.debug('////////////////////// Contour area done //////////////////////')
        self.history +=1
        if get_rects == True: # Return peoplerects without frame
            return occupied,  self.peopleRects 
        else:
            return occupied,  frame