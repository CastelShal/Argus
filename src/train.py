import os
import cv2
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from detector_modules.dlib_face_detector import DlibFaceDetector
from utils.face_aligner import align
from detector_modules.facenet import get_embedder_instance
TRAINING_DIR="./training"

face_detector = DlibFaceDetector()
embedder = get_embedder_instance()
training_df = {}

# detect face and generate embedding
def get_face_embedding(frame):
    detections, faces = face_detector.detectFaces(frame)
    if faces is not None and len(detections) > 0:
        aligned_faces = align(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
            detections
        )
        face = aligned_faces[0]
        if face is not None:
                return embedder.embeddings([face])
    return None

#   parse each dir seperately for image paths
def get_image_paths(dir_name):
    dirpath = os.path.join(TRAINING_DIR, dir_name)
    files = next(os.walk(dirpath))[2]
    image_paths = [os.path.join(dirpath, file) for file in files]
    return image_paths

#   read from paths and get embeds
def get_embeddings_from_images(image_paths):
    res = []
    for path in image_paths:
        img = cv2.imread(path)
        embedding = get_face_embedding(img)
        if embedding is not None:
            res.append(embedding[0].tolist())
    return res

dirs = next(os.walk(TRAINING_DIR))[1]

for dir_name in dirs:
    image_paths = get_image_paths(dir_name)
    embeddings = get_embeddings_from_images(image_paths)
    face_name = dir_name.capitalize()
    training_df[face_name] = embeddings.copy()
    print(f"Added embeddings for {face_name}")

#   convert to json
with open("src/data/op.json", "w") as file:
    json.dump(training_df, file)
