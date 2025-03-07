import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from nodes.node import Node

main = Node("videos/peopleTest.m4v")
main.process_capture()
