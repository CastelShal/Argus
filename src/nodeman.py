from nodes.streaming_node import StreamingNode
import threading

class NodeManager:
    def __init__(self, database):
        self.nodes = {}
        self.database = database

    def add_node(self, url):
        """Add a new node with a unique ID and associated data."""
        node_id = str(len(self.nodes) + 1)
        newNode = StreamingNode(url, self.database)
        stream_thread = threading.Thread(target=newNode.process_capture)
        stream_thread.daemon = True
        newNode.setStreamThread(stream_thread)
        stream_thread.start()

        self.nodes[node_id] = newNode

    def remove_node(self, node_id):
        """Remove an existing node by its ID."""
        if node_id not in self.nodes:
            print(f"Node with ID {node_id} does not exist.")
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
    
    def truncate(self):
        """Remove all nodes."""
        for node_id in list(self.nodes.keys()):
            self.remove_node(node_id)
        self.nodes.clear()