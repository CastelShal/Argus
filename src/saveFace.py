import os
import cv2
import pprint
import json

from models.dlib_face_detector import DlibFaceDetector
from utils.face_aligner import align
from keras_facenet import FaceNet

embedder = FaceNet()
face_detector = DlibFaceDetector()
dirs = next(os.walk("./training"))[1]
training_df = {}

def img_to_embedding(frame):
    global face_detector
    detections, faces = face_detector.detectFaces(frame, 0.75)
    res = []
    if faces is not None:
        if len(detections) > 0:
            aligned_faces = align(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            detections)
            face = aligned_faces[0]
            if face is not None:
                cv2.imshow("face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))   
                return embedder.embeddings([face])
    return None
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
    training_df[dir] = res.copy()
    res.clear()

pprint.PrettyPrinter().pprint(training_df)
with open("op.json", "w") as file:
    json.dump(training_df, file)
