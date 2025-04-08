import cv2
import keras_facenet
from thread_camera import ThreadedCamera
from models.dlib_face_detector import DlibFaceDetector
from utils.face_aligner import align
import time
import logging
import numpy as np
from utils.sendMail import send_alert

camLogger = logging.getLogger("CameraLogger")

def create_end_frame():
    ''' Creates a frame to be displayed when the camera feed ends '''
    frame = np.zeros((520, 800, 3), dtype=np.uint8)

    text = "Camera feed ended"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# Class that represents a processing node for a single camera
# This class will handle face detection and face recognition
class StreamingNode:
    def __init__(self, capture, database, name, alerts=False):
        # parameters and initialization
        self.cap = ThreadedCamera(capture)
        self.database = database
        self.cname = name 
        self.enableAlerts = alerts
        self.face_detector = DlibFaceDetector()
        self.embedder = keras_facenet.FaceNet()
        self.embedder.embeddings([cv2.imread("training/seed.jpg")])     # Warm up the model 

        # local data setup
        self.face_boxes = []
        self.face_chops = []
        self.alert = False
        self.unknown = 0
        self.processed = None
        self.streamThread = None
        self.timer = time.time()
        self.prev_found = set()
        self.found = set()
    
    def process_capture(self):
        while not self.cap.ready:
            pass
        while self.cap.running:
            frame = self.cap.read()
            if frame is None: break
            self.face_boxes.clear()

            # Pre-processing
            frame = cv2.resize(frame, (800, 520))

            ##### Facial Detection ######
            detections, faces = self.face_detector.detectFaces(frame)
            
            if faces is not None and len(detections) > 0:
                for bbox in faces:
                    x0, y0, x1, y1 = bbox
                    rect = (x0, y0, x1, y1)
                    self.face_boxes.append(rect)
                
                ######  Facial Alignment and Embedding  #######
                self.face_chops = align(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detections)
                embeddings = self.embedder.embeddings(self.face_chops)
                
                unknown_this_frame = False
                for query, bbox in zip(embeddings, self.face_boxes):
                    result = self.database.cosine_similarity_search(query)
                    x, y, r, b = bbox
                    if result["score"] > .5:
                        cv2.rectangle(frame, (x, y), (r, b), (0, 255, 0), 4)
                        cv2.putText(frame, result["name"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        camLogger.debug(f"{self.cname}- Matched:{result["name"]}-{result["score"]}")
                        self.found.add(result["name"])
                    else:
                        cv2.rectangle(frame, (x, y), (r, b), (255, 0, 0), 4)
                        camLogger.debug(f"{self.cname}-No match: Nearest {result['name']}-{result['score']}")
                        unknown_this_frame = True
                        self.unknown += 1

                ######   Logging processing    #######
                if (time.time() - self.timer) >= 1:
                    if len(self.found) > 0:
                        found_this_frame = self.found - self.prev_found
                        print(self.found, self.prev_found, found_this_frame)
                        if len(found_this_frame) > 0:
                            camLogger.info(f"{self.cname} - Found: {", ".join(found_this_frame)}")

                    self.prev_found = self.found.copy()
                    self.found.clear()
                    self.timer = time.time()

                #####    Alert processing    #####
                if not unknown_this_frame:
                    self.unknown = max(0, self.unknown - 1)     # Avoids temporary artifacts causing false positives
                if self.unknown > 5 and not self.alert:
                    camLogger.warning(f"-----UNKNOWN INDIVIDUALS DETECTED ON {self.cname}")
                    if self.enableAlerts: 
                        self.alert = True
                        send_alert(self.cname)
            
            self.processed = frame
        print(f"Camera feed {self.cname} terminated.")

    def gen_frames(self):
        while not self.cap.ready:
            print("Waiting for camera to be ready...")
            pass  
        while self.cap.running:
            frame = self.processed 
            if frame is None:
                break
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1/30)
        yield (b'--frame\r\n' 
            b'Content-Type: image/jpeg\r\n\r\n' + create_end_frame() + b'\r\n')

    def setStreamThread(self, thread):
        self.streamThread = thread
 