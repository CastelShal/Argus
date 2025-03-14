import keras_facenet
from thread_camera import ThreadedCamera
from models.motion_detector import MotionDetector
from models.dlib_face_detector import DlibFaceDetector
from utils.imgUtils import draw_rect
import cv2

# Class that represents a processing node for a single camera (for now)
# This class will handle the face detection, motion detection and face recognition
class MotionNode:
    DEBUG_MOTION_BOXES = False
    def __init__(self, capture):
        self.face_detector = DlibFaceDetector()
        self.motion_detector = MotionDetector()
        self.embedder = keras_facenet.FaceNet()
        self.cap = ThreadedCamera(capture)
        self.face_boxes = []
        self.motion_boxes = []
        self.cname = capture
    
    def process_capture(self):
        while not self.cap.ready:
            pass    # Wait for the camera to start, TO-DO: Add a timeout
        while self.cap.running:
            frame = self.cap.read()
            
            self.face_boxes.clear()
            self.motion_boxes.clear()

            _, rects = self.motion_detector.detect_movement(frame, True)
            for box in rects:
                (x, y, w, h) = box
                crop = frame[y: y+h, x: x+w]  # Crop the areas where motion was detected
                if self.DEBUG_MOTION_BOXES: self.motion_boxes.append((x, y, w, h))
                cx, cy, _ = crop.shape
                if cx == 0 or cy == 0: continue

                _, faces = self.face_detector.detectFaces(crop, 0.75)
                if faces is not None:
                    for bbox in faces:
                        x0, y0, x1, y1 = bbox
                        rect = (x0 + x, y0 + y, x1 + x, y1 + y)   # Adjust the coordinates to the original frame
                        self.face_boxes.append(rect)
            
            if self.DEBUG_MOTION_BOXES:
                for box in self.motion_boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)

            for bbox in self.face_boxes:
                draw_rect(frame, bbox)
            cv2.imshow(self.cname, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.stop()
        cv2.destroyWindow(self.cname)
        print(f"Camera feed {self.cname} terminated.")