import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from nodes.streaming_node import StreamingNode
from database import Database
import json
import threading
from flask import Flask, render_template, Response
#Initialize the Flask app

# Load the embeddings from the JSON file
data = json.load(open("op.json"))
vector_store = Database()
vector_store.populate_database(data)

app = Flask(__name__)

main = StreamingNode("videos/peopleTest.m4v", vector_store)
node = threading.Thread(target=main.process_capture)
node.daemon = True
node.start()

next = StreamingNode("videos/ppl.mp4", vector_store)
node2 = threading.Thread(target=next.process_capture)
node2.daemon = True
node2.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(main.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(next.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)