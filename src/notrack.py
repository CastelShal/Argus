import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import keras_facenet
from imgUtils import draw_rect
from mp_face_detector import MediaPipeDetector
from dlib_face_detector import DlibFaceDetector
from mtcnn_det import MTCNNDetector

detector = DlibFaceDetector()
embedder = keras_facenet.FaceNet()

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
          
      # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
      bboxes.clear()
      imgs.clear()

      faces = detector.detectFaces(frame, 0.75)
      if faces is not None:
        for bbox in faces:
          x0, y0, x1, y1 = bbox
          roi = frame[y0:y1, x0:x1]
          imgs.append(roi)
          rect = (x0, y0, x1, y1)
          bboxes.append(rect)

      for bbox in bboxes:
        draw_rect(frame, bbox)
      cv2.imshow(f"Camera {camera}", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyWindow(f"Camera {camera}")
  print(f"Camera feed {camera} terminated.")

process_camera_feed("videos/ppl.mp4")
# process_camera_feed("http://192.168.0.105:8080/video")
