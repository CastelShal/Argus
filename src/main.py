import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from nodes.node import Node
from database import Database
import numpy as np
import json

# Load the embeddings from the JSON file
data = json.load(open("op.json"))
vector_store = Database()
vector_store.populate_database(data)

main = Node("videos/peopleTest.m4v", vector_store)
main.process_capture()
