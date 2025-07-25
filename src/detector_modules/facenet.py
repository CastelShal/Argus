from keras_facenet import FaceNet
from cv2 import imread

FACENET_KEY = "20180408-102900"
CACHE_FOLDER = "models/"

SEED = imread("src/detector_modules/seed.jpg")
def get_embedder_instance():
    embedder = FaceNet(key=FACENET_KEY ,cache_folder=CACHE_FOLDER)
    embedder.embeddings([SEED])  # seed image to warm up the model
    return embedder