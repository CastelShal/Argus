import os
import pprint
import json
import numpy as np

dirs = next(os.walk("./training"))[1]

training_df = {}

for dir in dirs:
    dirpath = os.path.join("./training", dir)
    files = next(os.walk(dirpath))[2]
    file_paths = [os.path.join("./training", dir, file) for file in files]
    res = []
    for path in file_paths:
        with open(path, "r") as file:
            embed = np.random.random_sample(128) * 10
            embed = embed.tolist()
            res.append(embed)
    training_df[dir] = res.copy()
    res.clear()

pprint.PrettyPrinter().pprint(training_df)
with open("op.json", "w") as file:
    json.dump(training_df, file)

# import cv2
# from imgUtils import rgb_pre_processing, draw_rect
# from keras_facenet import FaceNet
# embedder = FaceNet()

# img = cv2.imread("jeff.jpeg")
# rgb = rgb_pre_processing(img)

# embed = embedder.extract(img, 0.8)
# box = embed[0]['box']
# box[2] += box[0]
# box[3] += box[1]
# draw_rect(img, box)
# cv2.imwrite("processed.jpeg", img)
# print(embed)
