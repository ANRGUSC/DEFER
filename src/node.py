import orjson as json
import queue
import socket
from threading import Thread
from queue import Queue
import select
import time

import numpy as np
import tensorflow as tf

from node_state import NodeState, socket_recv, socket_send

# port 5000 is data, 5001 is model, 5002 is weights

class Node:
    def model_socket(node_state: NodeState):
        model_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        model_server.bind(("0.0.0.0", 5001))
        print("Model socket running")
        model_server.listen(1) 
        model_cli = model_server.accept()[0]
        model_cli.setblocking(0)

        model_json = socket_recv(model_cli, node_state.chunk_size)
        # Getting address of next node
        next_node = socket_recv(model_cli, chunk_size=1)
        print(next_node.decode())
        
        part = tf.keras.models.model_from_json(model_json)
        while (node_state.weights == ""): # Waiting for weights to be sent on other thread
            time.sleep(5)
        part.set_weights(node_state.weights)
        id = socket.gethostbyname(socket.gethostname())
        # part.save(f'model_{id}')
        # md = tf.keras.models.load_model(f'model_{id}')
        md = part
        md._make_predict_function()
        node_state.model = md
        tf.keras.utils.plot_model(md, f"model_{id}.png")
        node_state.next_node = next_node.decode()
        print("Waiting to send acknowledgement")
        select.select([], [model_cli], [])
        model_cli.send(b'\x06')
        print("Sent acknowledgement")
        model_server.close()
        print("Model thread over")

    def weights_socket(node_state: NodeState):
        chunk_size = node_state.chunk_size
        weights_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        weights_server.bind(("0.0.0.0", 5002))
        print("Weights socket running")
        weights_server.listen(1)
        weights_cli = weights_server.accept()[0]
        print("Connection accepted")
        weights_cli.setblocking(0)

        weights_json = socket_recv(weights_cli, chunk_size)

        print("Deserializing json")
        w_list = json.loads(weights_json)
        print("json deserialized")
        model_weights = [np.array(w, dtype=np.float32) for w in w_list]
        print("Array found")
        node_state.weights = model_weights
        weights_server.close()
        print("Weights thread over")

    def data_server(node_state: NodeState, to_send: Queue):
        chunk_size = node_state.chunk_size
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_server.bind(("0.0.0.0", 5000))
        print("Data socket running")
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setblocking(0)

        while True:
            data_json = socket_recv(data_cli, chunk_size)

            inpt = np.array(json.loads(data_json))
            #print("Sending to client queue")
            to_send.put(inpt)

    def data_client(node_state: NodeState, to_send: Queue):
        graph = tf.get_default_graph()
        while node_state.next_node == "":
            time.sleep(5)# Wait until next_node is set by model socket
        chunk_size = node_state.chunk_size
        model = node_state.model
        next_node_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Data client running")
        next_node_client.connect((node_state.next_node, 5000))
        next_node_client.setblocking(0)
        
        while True:
            inpt = to_send.get()
            with graph.as_default():
                output = model.predict(inpt).tolist()
            out_json = json.dumps(output)
            socket_send(out_json, next_node_client, chunk_size)
            #print("Sent result to node:", node_state.next_node)

    def run(self):
        ns = NodeState(chunk_size = 512 * 1000)
        m = Thread(target=self.model_socket, args=(ns,))
        w = Thread(target=self.weights_socket, args=(ns,))
        to_send = queue.Queue(1000) # Arbitrary size of queue, can change later
        dserv = Thread(target=self.data_server, args=(ns, to_send))
        dcli = Thread(target=self.data_client, args=(ns, to_send))
        m.start()
        w.start()
        dserv.start()
        dcli.start()
        m.join()
        w.join()
        dserv.join()
        dcli.join()

node = Node()
node.run()

