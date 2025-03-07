import dlib
import cv2
import time
predictor_path = "./shape_predictor_5_face_landmarks.dat"
face_file_path = "./face.jpg"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Load the image using Dlib
img = cv2.imread(face_file_path)
dets = detector(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 1)
num_faces = len(dets)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()

# Find the 5 face landmarks we need to do the alignment.
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))

# Get the aligned face images
# Optionally: 
# images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
images = dlib.get_face_chips(img, faces, size=320)
for image in images:
    cv2.imshow("Aligned Face", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1000)
    dlib.hit_enter_to_continue()


