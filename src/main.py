import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import keras_facenet
from thread_camera import ThreadedCamera
from motion_detector import MotionDetector

from utils.imgUtils import draw_rect
from models.dlib_face_detector import DlibFaceDetector

detector = DlibFaceDetector()
embedder = keras_facenet.FaceNet()
motion_detector = MotionDetector()

def process_camera_feed(camera):
  global detector
  global motion_detector
  # cap = cv2.VideoCapture(camera)
  
  # if not cap.isOpened():
    # print(f"Failed to open camera {camera}")
    # return

  cap = ThreadedCamera(camera)
  face_boxes = []
  imgs = []
  while not cap.ready:
    pass
  while True:
      ret, frame = cap.read()
      if not ret:
          print(f"Failed to grab frame from camera {camera}")
          break
      # frame = cv2.resize(frame, (640, 480))
      
      face_boxes.clear()
      imgs.clear()

      # faces = detector.detectFaces(frame, 0.75)
      # if faces is not None:
      #   for bbox in faces:
      #     x0, y0, x1, y1 = bbox
      #     roi = frame[y0:y1, x0:x1]
      #     imgs.append(roi)
      #     rect = (x0, y0, x1, y1)
      #     face_boxes.append(rect)
      _, rects = motion_detector.detect_movement(frame, True)
      for box in rects:
        (x, y, w, h) = box
        crop = frame[y: y+h, x: x+w]
        imgs.append((x, y, w, h))
        cx, cy, cz = crop.shape

        if cx == 0 or cy == 0: continue
        faces = detector.detectFaces(crop, 0.75)
        if faces is not None:
          for bbox in faces:
            x0, y0, x1, y1 = bbox
            x0 += x
            y0 += y
            x1 += x
            y1 += y
            rect = (x0, y0, x1, y1)
            face_boxes.append(rect)

        # face_boxes.append((x, y, x + w, y + h))
      
      for img in imgs:
         cv2.rectangle(frame, (img[0], img[1]), (img[0] + img[2], img[1] + img[3]), (0, 0, 255), 2)
      for bbox in face_boxes:
        draw_rect(frame, bbox)
      cv2.imshow(f"Camera {camera}", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.stop()
  cv2.destroyAllWindows()
  print(f"Camera feed {camera} terminated.")

process_camera_feed("videos/peopleTest.m4v")
# process_camera_feed("http://192.168.0.105:8080/video")
