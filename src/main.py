import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow GPU warnings

from webapp.webapp import setup_webapp
from data.data_store import Database
from manager import NodeManager
import utils.logger as logger

logger.setLoggers()

data = json.load(open("src/data/op.json"))
vector_store = Database()
vector_store.populate_database(data)

node_manager = NodeManager(vector_store)
app = setup_webapp(node_manager)
app.run(host="0.0.0.0", debug=False)
