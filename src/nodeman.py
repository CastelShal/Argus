from nodes.streaming_node import StreamingNode
import threading
import logging

adminLogger = logging.getLogger("AdminLogger")
class NodeManager:
    def __init__(self, database):
        self.nodes = {}
        self.database = database

    def add_node(self, url, name, alerts=False, node_id=None):
        """Add a new node with a unique ID and associated data."""
        nid = node_id or str(len(self.nodes) + 1)
        newNode = StreamingNode(url, self.database, name, alerts)

        stream_thread = threading.Thread(target=newNode.process_capture)
        stream_thread.daemon = True
        newNode.setStreamThread(stream_thread)
        stream_thread.start()

        self.nodes[nid] = newNode
        adminLogger.info(f"Node Addition - ID: {nid}")

    def upsert_node(self, node_id, url, name, alerts=False):
        """Update an existing node or add a new one if it doesn't exist."""

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
            self.add_node(url, name, alerts)

    def remove_node(self, node_id):
        """Remove an existing node by its ID."""

        if node_id not in self.nodes:
            adminLogger.warning(f"Node {node_id} not found for removal.")
            return 
        
        delNode = self.nodes[node_id]
        delNode.cap.stop()
        del self.nodes[node_id]

    def get_node(self, node_id):
        """Retrieve a node by its ID."""
        
        return self.nodes.get(node_id)
    
    def check_alert(self, node_id):
        """Check if an alert has been raised for a node."""

        node = self.get_node(node_id)
        if not node:
            return None
        return node.alert

    def get_all_nodes(self):
        """Retrieve all nodes."""

        return self.nodes
    
    def truncate(self, silent=False):
        """Remove all nodes."""

        if len(self.nodes) > 0:
            if not silent: adminLogger.info("Clearing all nodes.")
            for node_id in list(self.nodes.keys()):
                self.remove_node(node_id)