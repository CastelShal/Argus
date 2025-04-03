import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import json
from database import Database
from nodeman import NodeManager
from flask import Flask, render_template, Response, request
import logger
logger.setLoggers()

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
    # node_manager.truncate(silent=True) 
    data = request.json
    if not data or 'nodes' not in data:
        return "Invalid request. 'nodes' is required.", 400
    nodes = data['nodes']
    if not isinstance(nodes, list):
        return "'nodes' must be a list.", 400
    
    for node in nodes:
        if 'url' not in node:
            return "Each node must have a 'url'.", 400
        node_manager.upsert_node(
            node_id=node.get('node_id', ''),
            url=node['url'],
            name=node.get('name'),
            alerts=node.get('alerts', False)
        )
    
    return "Nodes updated successfully.", 200

@app.route("/dashboard")
def dashboard():
    nodes = node_manager.get_all_nodes()  # Get all node IDs

    return render_template("dashboard.html", nodes=nodes)

@app.route("/config")
def config():
    nodes = node_manager.get_all_nodes()
    # Transform node data into the format expected by the template
    formatted_nodes = [
        {
            "id": node_id,  # Include the node_id
            "url": node.cap.url,  # Assuming this is the URL
            "name": node.cname,   # Assuming this is the name
            "alerts": node.enableAlerts,  # Assuming this is the alerts flag
        }
        for node_id, node in nodes.items()
    ]
    return render_template("config.html", nodes=formatted_nodes)

app.run(debug=True)