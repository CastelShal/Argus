import cv2
import keras_facenet
from thread_camera import ThreadedCamera
from models.dlib_face_detector import DlibFaceDetector
from utils.imgUtils import draw_rect
from utils.face_aligner import align

# Class that represents a processing node for a single camera (for now)
# This class will handle face detection and face recognition
class Node:
    def __init__(self, capture, database):
        self.face_detector = DlibFaceDetector()
        self.embedder = keras_facenet.FaceNet()
        self.cap = ThreadedCamera(capture)
        self.face_boxes = []
        self.face_chops = []
        self.cname = capture
        self.database = database
    
    def process_capture(self):
        rand = 0
        while not self.cap.ready:
            pass    # Wait for the camera to start, TO-DO: Add a timeout
        while True:
            frame = self.cap.read()
            if frame is None: continue
            self.face_boxes.clear()
            detections, faces = self.face_detector.detectFaces(frame, 0.75)
            
            if faces is not None and len(detections) > 0:
                
                for bbox in faces:
                    x0, y0, x1, y1 = bbox
                    rect = (x0, y0, x1, y1)   # Adjust the coordinates to the original frame
                    self.face_boxes.append(rect)

                for bbox in self.face_boxes:
                    draw_rect(frame, bbox)
                
                self.face_chops = align(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detections)
                embeddings = self.embedder.embeddings(self.face_chops)
                
                for query, bbox in zip(embeddings, self.face_boxes):
                    result = self.database.cosine_similarity_search(query)
                    
                    if result["score"] > .6:
                        x, y, r, b = bbox
                        cv2.rectangle(frame, (x, y), (r, b), (]), 4)
                        print(f"Match found: {result["name"]} - score {result["score"]}")
                    else:
                        print("No match found. Highest candidate", result)
            
            cv2.imshow(self.cname, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.stop()
        cv2.destroyWindow(self.cname)
        print(f"Camera feed {self.cname} terminated.")