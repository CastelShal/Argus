import time
class Person:
    def __init__(self, id, bbox, embedding, color = (0,0,0)):
        self.id = id
        self.bbox = bbox
        self.color = color
        self.time = time.time()
        self.embedding = embedding

    def reset(self):
        self.time = time.time()