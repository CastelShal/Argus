import mediapipe as mp

class MediaPipeDetector:
    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0)

    def detectFaces(self, frame, threshold = 0.7):
        res = self.detector.process(frame)
        if res.detections is None:
            return None
        
        bboxes = []
        for det in res.detections:
            if det.score[0] > threshold:
                box = self.get_bbox(det, frame.shape[1], frame.shape[0])
                if box is not None:
                    bboxes.append(box)
        return bboxes

    def get_bbox(self, detection, w, h):
        face = detection.location_data.relative_bounding_box
        xmin, ymin, width, height = face.xmin, face.ymin, face.width, face.height
        if xmin < 0 or ymin < 0 or width < 0 or height < 0:
            return None
        
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        width = int(w * width)
        height = int(h * height)
        return [xmin, ymin, xmin + width, ymin + height]

