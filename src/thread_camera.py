import cv2
import time
from threading import Thread

class ThreadedCamera(Thread):
    def __init__(self, src):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.url = src
        self.frame = None
        self.ret = False
        self.running = True
        self.ready = False
        self.daemon = True
        self.start()

    def run(self):
        (self.ret, self.frame) = self.capture.read()
        if not self.ret:
            print(f'Capture url {self.url} is invalid')
            raise FileNotFoundError
        self.ready = True

        while self.ret:
            if self.capture.isOpened():
                (self.ret, self.frame) = self.capture.read()
            else:
                print("Camera failed to run")
                break
            time.sleep(1/22)
        self.capture.release()
        self.stop()

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
