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

import lz4.frame
import zfpy
import time

# 5000 is data, 5001 is model architecture, 5002 is weights

class DEFER:
    def __init__(self, computeNodes) -> None:
        self.computeNodes = computeNodes
        self.dispatchIP = socket.gethostbyname(socket.gethostname())
        self.chunk_size = 512 * 1000
        self.graph = tf.get_default_graph()
    
    def _partition(self, model: tf.keras.Model, layer_parts: List[str]) -> List[tf.keras.Model]:
        with self.graph.as_default():
            models = []
            for p in range(len(layer_parts) + 1):
                if p == 0:
                    start = model.input._keras_history[0].name
                else:
                    start = layer_parts[p-1]
                if p == len(layer_parts):
                    print(model.output)
                    end = model.output._keras_history[0].name
                else:
                    end = layer_parts[p]
                part = construct_model(model, start, end, part_name=f"part{p+1}")
                models.append(part)
            return models

    def _dispatchModels(self, models: list, nodeIPs: List[str]) -> None:
        for i in range(len(models)):
            weights_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            weights_sock.setblocking(0)
            weights_sock.settimeout(10)
            model_json = models[i].to_json()
            weights_sock.connect((nodeIPs[i], 5002))
            if i != len(models) - 1:
                nextNode = nodeIPs[i + 1]
            else:
                # Reached the end of the nodes, the last node needs to point back to the dispatcher
                nextNode = self.dispatchIP
            
            self._send_weights(models[i].get_weights(), weights_sock, self.chunk_size)
            model_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            model_sock.setblocking(0)
            model_sock.settimeout(10)
            model_sock.connect((nodeIPs[i], 5001))
            socket_send(model_json.encode(), model_sock, self.chunk_size)
            socket_send(nextNode.encode(), model_sock, chunk_size=1)
            select.select([model_sock], [], []) # Waiting for acknowledgement
            model_sock.recv(1)

    def _send_weights(self, weights: List, sock: socket.socket, chunk_size: int):
        size = len(weights)
        size_bytes = size.to_bytes(8, 'big')
        while len(size_bytes) > 0:
            try:
                sent = sock.send(size_bytes)
                size_bytes = size_bytes[sent:]
            except socket.error as e:
                    if e.errno != socket.EAGAIN:
                        raise e
                    select.select([], [sock], [])
        for w_arr in weights:
                as_bytes = self._comp(w_arr)
                socket_send(as_bytes, sock, chunk_size)
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        return zfpy.compress_numpy(lz4.frame.decompress(byts))
    def _startDistEdgeInference(self, input: queue.Queue):
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_sock.connect((self.computeNodes[0], 5000))
        data_sock.setblocking(0)
        
        while True:
            model_input = input.get()
            out = self._comp(model_input)
            socket_send(out, data_sock, self.chunk_size)

    def _result_server(self, output: queue.Queue):
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        data_server.bind(("0.0.0.0", 5000)) 
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setblocking(0)

        while True:
            data = bytes(socket_recv(data_cli, self.chunk_size))
            pred = self._decomp(data)
            output.put(pred)

    def run_defer(self, model: tf.keras.Model, partition_layers, input_stream: queue.Queue, output_stream: queue.Queue):
        models_to_dispatch = self._partition(model, partition_layers)
        a = threading.Thread(target=self._result_server, args=(output_stream,))
        a.start()
        self._dispatchModels(models_to_dispatch, self.computeNodes)
        time.sleep(2)
        b = threading.Thread(target=self._startDistEdgeInference, args=(input_stream,), daemon=True)
        b.start()
        a.join()


