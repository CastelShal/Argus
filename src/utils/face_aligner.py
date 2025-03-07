import dlib

predictor_path = "src/models/shape_predictor_5_face_landmarks.dat"
shaper = dlib.shape_predictor(predictor_path)

def align(img, dets):
    """Aligns images that were put through the face detector. Pass dets as the raw output of dlib's detector"""
    detection_store = dlib.full_object_detections()
    for detection in dets:
        detection_store.append(shaper(img, detection))

    images = dlib.get_face_chips(img, detection_store)
    return images