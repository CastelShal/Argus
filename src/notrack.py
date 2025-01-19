import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from numpy import linalg as la
import numpy as np
import mediapipe as mp
import random
import time
import keras_facenet
from person import Person
from imgUtils import draw_rect, get_bbox, rgb_pre_processing

mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

embedder = keras_facenet.FaceNet()
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
            roi = rgb[ymin: ymin+height, xmin:xmin+width]
            imgs.append(roi)
            rect = (xmin, ymin, xmin + width, ymin + height)
            bboxes.append(rect)
      # cleanup_face_db()
      if len(imgs) > 0:
        process_similarity(bboxes, imgs)

      for person in this_frame:
        draw_rect(frame, person.bbox, color=person.color)
      cv2.imshow(f"Camera {camera}", frame)

      if cv2.waitKey(5) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyWindow(f"Camera {camera}")
  print(f"Camera feed {camera} terminated.")

def get_min_dist(embed):
  threshold = 0.3
  global face_db
  candid = None
  min_dist = threshold
  for person in face_db:
    dist = embedder.compute_distance(person.embedding, embed)
    print(dist)
    if min_dist > dist:
        min_dist = dist
        candid = person
  return (candid if min_dist < threshold else None)

def process_similarity(bboxes, imgs):
  global face_db
  global person_id
  global embedder
  global this_frame
  embeddings = embedder.embeddings(imgs)
  for bbox, embed in zip(bboxes, embeddings):
    candid = get_min_dist(embed)
    print(candid)
    if candid is not None:
      candid.embedding = embed
      candid.bbox = bbox
      candid.reset()
      this_frame.append(candid)
    else:
      person = Person( person_id, bbox, embed, (random.randint(0, 255), random.randint(0,255), random.randint(0, 255)) )
      person_id += 1
      face_db.append(person)
      this_frame.append(person)


def cleanup_face_db():
   for person in face_db:
      if time.time() - person.time > 1.5:
         face_db.remove(person)

def find_faces_or_store( face_embeds, bboxes ):
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

process_camera_feed("../videos/peopleTest.m4v")
# process_camera_feed("http://192.168.0.105:8080/video")
