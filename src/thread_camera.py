import cv2
import time
from threading import Thread

class ThreadedCamera(Thread):
    def __init__(self, src):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.frame = None
        self.ret = False
        self.running = True
        self.ready = False
        self.start()

    def run(self):
        while self.running:
            if self.capture.isOpened():
                (self.ret, self.frame) = self.capture.read()
                self.ready = True
            else:
                print("Camera failed to run")
                break
            time.sleep(1/22)
        self.capture.release()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
