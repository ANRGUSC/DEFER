import orjson as json
import select
import socket
import threading
from typing import List
import queue

import numpy as np
import tensorflow as tf

from dag_util import *
from node_state import socket_recv, socket_send

# 5000 is data, 5001 is model, 5002 is weights

class DEFER:
    def __init__(self, computeNodes, ) -> None:
        self.computeNodes = computeNodes
        # num_nodes = 8
        # computeNodes = [f"10.0.0.{i}" for i in range(21, 21 + num_nodes)]
        self.dispatchIP = socket.gethostbyname(socket.gethostname())
        self.chunk_size = 512 * 1000
    
    def partition(model: tf.keras.Model, layer_parts: List[str]) -> List[tf.keras.Model]:
        models = []
        for p in range(len(layer_parts) + 1):
            if p == 0:
                start = model.input._keras_history[0].name
            else:
                start = layer_parts[p-1]
            if p == len(layer_parts):
                end = model.output._keras_history[0].name
            else:
                end = layer_parts[p]
            part = construct_model(model, start, end, part_name=f"part{p+1}")
            models.append(part)
        return models

    def dispatchModels(self, models: list, nodeIPs: List[str]) -> None:
        for i in range(len(models)):
            weights_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            weights_sock.setblocking(0)
            weights_sock.settimeout(10)
            model_json = models[i].to_json()
            print("Current node:", nodeIPs[i])
            weights_sock.connect((nodeIPs[i], 5002))
            print("Connected to weights socket")
            if i != len(models) - 1:
                nextNode = nodeIPs[i + 1]
            else:
                # Reached the end of the nodes, the last node needs to point back to the dispatcher
                nextNode = self.dispatchIP
            as_list = [l.tolist() for l in models[i].get_weights()]
            print("weights list")
            as_json = json.dumps(as_list)
            print("JSON dumped")
            print("Sending weights")
            socket_send(as_json, weights_sock, self.chunk_size)

            print("Sending model")
            model_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            model_sock.setblocking(0)
            model_sock.settimeout(10)
            model_sock.connect((nodeIPs[i], 5001))
            socket_send(model_json.encode(), model_sock, self.chunk_size)
            socket_send(nextNode.encode(), model_sock, chunk_size=1)
            print("Sent model and next node")
            select.select([model_sock], [], []) # Waiting for acknowledgement
            model_sock.recv(1)
            print("Got acknowledgement")

    def startDistEdgeInference(self, input: queue.Queue):
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Starting inference")
        data_sock.connect((self.computeNodes[0], 5000))
        data_sock.setblocking(0)
        
        while True:
            model_input = input.get()
            out_json = json.dumps(model_input.tolist())
            socket_send(out_json, data_sock, self.chunk_size)

    def result_server(self, output: queue.Queue):
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        data_server.bind(("0.0.0.0", 5000)) 
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setblocking(0)

        while True:
            data_json = socket_recv(data_cli, self.chunk_size)

            pred = np.array(json.loads(data_json))
            output.put(pred)

    def run_defer(self, model, partition_layers, input_stream: queue.Queue, output_stream: queue.Queue):
        models_to_dispatch = self.partition(model, partition_layers)
        a = threading.Thread(target=self.result_server, args=(output_stream,))
        a.start()
        self.dispatchModels(models_to_dispatch, self.computeNodes)
        b = threading.Thread(target=self.startDistEdgeInference, args=(input_stream), daemon=True)
        b.start()
        a.join()


