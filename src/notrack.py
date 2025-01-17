import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from numpy import linalg as la
import numpy as np
import mediapipe as mp
import random
import time
from person import Person
from imgUtils import draw_rect, get_bbox, rgb_pre_processing

mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# embedder = keras_facenet.FaceNet()
face_db = []
this_frame = []
person_id = 0

# Function to process frames from a single camera feed
def process_camera_feed(camera):
  global detector
  cap = cv2.VideoCapture(camera)
  if not cap.isOpened():
    print(f"Failed to open camera {camera}")
    return

  bboxes = []
  imgs = []

  while True:
      ret, frame = cap.read()
      if not ret:
          print(f"Failed to grab frame from camera {camera}")
          break
      
      rgb = rgb_pre_processing(frame)
      # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      bboxes.clear()
      imgs.clear()
      this_frame.clear()

      faces = detector.process(rgb)
      h,w = rgb.shape[:2]
      if faces.detections is not None:
        for det in faces.detections:
          if det.score[0] < .7:
              continue
          
          bounding = get_bbox(det, w, h)
          if bounding is not None:
            xmin, ymin, width, height = get_bbox(det, w, h)
          #   roi = rgb[ymin: ymin+height, xmin:xmin+width]
          #   imgs.append(roi)
            rect = (xmin, ymin, xmin + width, ymin + height)
            this_frame.append(find_or_store_face(rect))
      cleanup_face_db()

      for person in this_frame:
        draw_rect(frame, person.bbox, color=person.color)
      cv2.imshow(f"Camera {camera}", frame)

      if cv2.waitKey(5) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyWindow(f"Camera {camera}")
  print(f"Camera feed {camera} terminated.")

def find_or_store_face(bbox):
  global face_db;
  global person_id;
  
  for person in face_db:
    dist = euclidean_distance(person.bbox, bbox)
    if dist < 25 and person.time - time.time() < 0.5:
      person.bbox = bbox
      person.reset()
      return person
    elif dist < 60:
        person.bbox = bbox
        person.reset()
        return person

  else:
    person = Person(person_id, bbox, (random.randint(0, 255), random.randint(0,255), random.randint(0, 255)))
    person_id += 1
    face_db.append(person)
    return person

def euclidean_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.linalg.norm(np.array(center1) - np.array(center2))

def cleanup_face_db():
   for person in face_db:
      if time.time() - person.time > 1.5:
         face_db.remove(person)

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
      # print("new dude")
      new_face = {"embed":np.array(f_embed), "color": (random.randint(0, 255), 0, random.randint(0, 255))}
      face_db.append(new_face)
      res.append(new_face["color"])
  return zip(bboxes, res)

process_camera_feed("../videos/ppl.mp4")
# process_camera_feed("http://192.168.0.105:8080/video")
