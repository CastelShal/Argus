import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from database import Database
import json
from nodeman import NodeManager
from flask import Flask, render_template, Response, request

# Load the embeddings from the JSON file
data = json.load(open("op.json"))
vector_store = Database()
vector_store.populate_database(data)

node_manager = NodeManager(vector_store)

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<node_id>')
def video_feed(node_id):
    node = node_manager.get_node(node_id)
    if not node:
        return f"Node with ID {node_id} not found", 404
    return Response(node.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_check/<node_id>')
def alert_check(node_id):
    alert = node_manager.check_alert(node_id)
    if alert is None:
        return f"Node with ID {node_id} not found", 404
    return str(alert)

@app.route('/shutdown')
def shutdown():
    node_manager.truncate()
    return "All nodes have been shut down."

@app.route('/add_nodes', methods=['POST'])
def add_nodes():
    node_manager.truncate()
    data = request.json
    if not data or 'video_paths' not in data:
        return "Invalid request. 'video_paths' is required.", 400
    video_paths = data['video_paths']
    if not isinstance(video_paths, list):
        return "'video_paths' must be a list.", 400
    for video_path in video_paths:
        node_manager.add_node(video_path)
    return "Nodes added successfully.", 200

@app.route("/dashboard")
def dashboard():
    node_ids = list(node_manager.get_all_nodes().keys())  # Get all node IDs
    print(node_ids)
    return render_template("dashboard.html", nodes=node_ids)

app.run()