import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import os
import cv2
# import pprint
import json

from models.dlib_face_detector import DlibFaceDetector
from utils.face_aligner import align
from keras_facenet import FaceNet

def img_to_embedding(frame):
    face_detector = DlibFaceDetector()
    detections, faces = face_detector.detectFaces(frame)
    if faces is not None:
        if len(detections) > 0:
            aligned_faces = align(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                detections
            )
            face = aligned_faces[0]
            if face is not None:
                return embedder.embeddings([face])
    return None

embedder = FaceNet()
dirs = next(os.walk("./training"))[1]
training_df = {}

for dir in dirs:
    dirpath = os.path.join("./training", dir)
    files = next(os.walk(dirpath))[2]
    file_paths = [os.path.join("./training", dir, file) for file in files]
    res = []
    for path in file_paths:
        img = cv2.imread(path)
        embedding = img_to_embedding(img)
        if embedding is not None:
            res.append(embedding[0].tolist())
    
    person_name = dir.capitalize()
    training_df[person_name] = res.copy()
    print(f"Added embeddings for {person_name}")
    res.clear()

# pprint.PrettyPrinter().pprint(training_df)
with open("src/data/op.json", "w") as file:
    json.dump(training_df, file)
