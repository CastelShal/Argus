import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from numpy import linalg as la
import numpy as np
import mediapipe as mp
import random
import keras_facenet
import time
mp_face_detection = mp.solutions.face_detection

embedder = keras_facenet.FaceNet()
face_db = []
timenow = time.time()
id = 0

# Function to process frames from a single camera feed
def process_camera_feed(camera_id, embeddings_db=[], threshold=0.4):
    cap = cv2.VideoCapture(camera_id)
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    if not cap.isOpened():
      print(f"Failed to open camera {camera_id}")
      return

    print(f"Processing camera feed {camera_id}...")

    # Variable to track face region (bounding box)
    bboxes = []
    imgs = []
    tracking = False 
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_id}")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not tracking:
            bboxes.clear()
            imgs.clear()
            faces = face_detection.process(rgb)
            h,w = rgb.shape[:2]
            
            if faces.detections is not None:
              for det in faces.detections:
                # if det.score[0] < 0.50:
                #   continue
                bounding = get_bbox(det, w, h)
                if bounding is not None:
                  xmin, ymin, width, height = get_bbox(det, w, h)
                  roi = rgb[ymin: ymin+height, xmin:xmin+width]
                  imgs.append(roi)

                  # rect = dlib.rectangle(xmin, ymin, xmin+width, ymin + height)
                  rect = (xmin, ymin, xmin+width, ymin + height)
                  bboxes.append(rect)
                  draw_rect(rgb, rect)
                  # tracker = cv2.TrackerKCF_create() #dlib.correlation_tracker()
                  # tracker.init(frame, rect)
                  # tracking = True                
                else:
                   continue     
        # else:
        #   face_bbox = update_tracker(frame, tracker)
        #   if face_bbox is None:
        #       print("tracking lost")
        #       tracking = False
        #   else:
        #       draw_rect(rgb, rect)

        # if len(bboxes) > 0 and len(imgs) > 0:
        #   try:
        #     face_embeds = embedder.embeddings(imgs)
        #   except Exception as e:
        #     print(f'There were {len(face_db)} faces')
        #     print(f'Died handling {len(imgs)} faces')
        #     print(imgs)
        #     print(bboxes[0])
        #     print(e)
        #     exit()
        #   boxes = find_faces_or_store(face_embeds, bboxes)

          # for box in boxes:
          #   bbox, color = box
          #   draw_rect(rgb, bbox, color)
        cv2.imshow(f"Camera {camera_id}", rgb)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {camera_id}")
    print(f"Camera feed {camera_id} terminated.")

def get_bbox(detection, w, h):
  face = detection.location_data.relative_bounding_box
  xmin, ymin, width, height = face.xmin, face.ymin, face.width, face.height
  if xmin < 0 or ymin < 0 or width < 0 or height < 0:
     print("Faulty face box")
     return None
  
  xmin = int(xmin * w)
  ymin = int(ymin * h)
  width = int(w * width)
  height = int(h * height)
  return [xmin, ymin, width, height]
   
def draw_rect(frame, bbox, color=(0, 255, 0)):
  # x, y, r, b = int(bbox.left()), int(bbox.top()), int(bbox.right()), int(bbox.bottom())
  x, y, r, b = bbox
  cv2.rectangle(frame, (x, y), (r, b), color, 4)

def find_faces_or_store(face_embeds, bboxes):
  '''
  Checks a face against all the faces in the db. If not found, stores it in. Returns a bbox, color pair.
  '''
  res = []  
  for f_embed in face_embeds:
    for embed in face_db:
      distance = la.norm(np.array(f_embed) - embed["embed"])
      if distance >= 0.2:
        res.append(embed["color"])
        break
    else:
      print("new dude")
      new_face = {"embed":np.array(f_embed), "color": (random.randint(0, 255), 0, random.randint(0, 255))}
      face_db.append(new_face)
      res.append(new_face["color"])
  return zip(bboxes, res)

def update_tracker(frame, tracker):
  success, face_bbox = tracker.update(frame)
  if not success:
      return None
  return face_bbox
  
process_camera_feed("video.mp4")
# process_camera_feed("http://192.168.0.105:8080/video")
