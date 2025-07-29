from streaming_node import StreamingNode
import threading
import logging

adminLogger = logging.getLogger("AdminLogger")
class NodeManager:
    def __init__(self, database):
        self.nodes = {}
        self.database = database

    def setup_stream_node(self, name, url, enableAlerts):
        node = StreamingNode(url, self.database, name, enableAlerts)
        stream_thread = threading.Thread(target=node.process_capture)
        stream_thread.daemon = True
        node.setStreamThread(stream_thread)
        stream_thread.start()
        return node

    def add_node(self, url, name, alerts=False, node_id=None):
        if node_id is None:
            node_id = str(len(self.nodes) + 1)  # auto generate ID if none provided

        newNode = self.setup_stream_node(name, url, enableAlerts=alerts)
        self.nodes[node_id] = newNode
        adminLogger.info(f"Node Addition - ID: {node_id}")

    def upsert_node(self, node_id, url, name, alerts=False):
        if node_id in self.nodes:
            adminLogger.info(f"Node Updated - ID: {node_id} with URL: {url}")

            if self.nodes[node_id].cap.url != url:
                self.remove_node(node_id)
                self.add_node(url, name, alerts)
            else:
                self.nodes[node_id].cname = name
                self.nodes[node_id].enableAlerts = alerts
            self.nodes[node_id].cname = name
            self.nodes[node_id].enableAlerts = alerts
        
        else:
            try:
                self.add_node(url, name, alerts)
            except FileNotFoundError:
                print(f'Cannot add node with url: ', url) 

    def remove_node(self, node_id):
        if node_id not in self.nodes:
            adminLogger.warning(f"Node {node_id} not found for removal.")
            return 
        
        delNode = self.nodes[node_id]
        delNode.cap.stop()
        del self.nodes[node_id]

    def get_node(self, node_id):
        return self.nodes.get(node_id)
    
    def check_alert(self, node_id):
        node = self.get_node(node_id)
        if not node:
            return None
        return (node.alert, node.cap.running)

    def get_all_nodes(self):
        return self.nodes
    
    def truncate(self, silent=False):
        if len(self.nodes) > 0:
            if not silent: adminLogger.info("Clearing all nodes.")
            for node_id in list(self.nodes.keys()):
                self.remove_node(node_id)