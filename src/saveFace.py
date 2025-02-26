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

