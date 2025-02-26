import cv2
import dlib

class DlibFaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detectFaces(self, frame, threshold=0.7):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector(gray, 1)
        bboxes = []
        for det in detections:
            box = self.get_bbox(det, frame.shape[1], frame.shape[0])
            if box is not None:
                bboxes.append(box)
        return bboxes

    def get_bbox(self, detection, w, h):
        xmin, ymin, xmax, ymax = detection.left(), detection.top(), detection.right(), detection.bottom()
        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
            return None
        
        return [xmin, ymin, xmax, ymax]
