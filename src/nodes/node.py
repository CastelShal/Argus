import cv2
import keras_facenet
from thread_camera import ThreadedCamera
from models.dlib_face_detector import DlibFaceDetector
from utils.imgUtils import draw_rect
from utils.face_aligner import align

# Class that represents a processing node for a single camera (for now)
# This class will handle face detection and face recognition
class Node:
    def __init__(self, capture):
        self.face_detector = DlibFaceDetector()
        self.embedder = keras_facenet.FaceNet()
        self.cap = ThreadedCamera(capture)
        self.face_boxes = []
        self.cname = capture
    
    def process_capture(self):
        while not self.cap.ready:
            pass    # Wait for the camera to start, TO-DO: Add a timeout
        while True:
            frame = self.cap.read()
            self.face_boxes.clear()

            detections, faces = self.face_detector.detectFaces(frame, 0.75)
            
            if faces is not None:
                if len(detections) > 0:
                    aligned_faces = align(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    detections)

                    for face in aligned_faces:
                        cv2.imwrite("aface.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                for bbox in faces:
                    x0, y0, x1, y1 = bbox
                    rect = (x0, y0, x1, y1)   # Adjust the coordinates to the original frame
                    self.face_boxes.append(rect)

                for bbox in self.face_boxes:
                    draw_rect(frame, bbox)
                cv2.imshow(self.cname, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.stop()
        cv2.destroyWindow(self.cname)
        print(f"Camera feed {self.cname} terminated.")