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

import zfpy
import lz4.frame

# port 5000 is data, 5001 is model architecture, 5002 is weights

class Node:
    def _model_socket(self, node_state: NodeState):
        model_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        model_server.bind(("0.0.0.0", 5001))
        print("Model socket running")
        model_server.listen(1) 
        model_cli = model_server.accept()[0]
        model_cli.setblocking(0)

        model_json = socket_recv(model_cli, node_state.chunk_size)
        next_node = socket_recv(model_cli, chunk_size=1)
        
        part = tf.keras.models.model_from_json(model_json)
        while (node_state.weights == ""): # Waiting for weights to be sent on other thread
            time.sleep(5)
        part.set_weights(node_state.weights)
        id = socket.gethostbyname(socket.gethostname())
        md = part
        md._make_predict_function()
        node_state.model = md
        tf.keras.utils.plot_model(md, f"model_{id}.png")
        node_state.next_node = next_node.decode()
        select.select([], [model_cli], [])
        model_cli.send(b'\x06')
        model_server.close()

    def _weights_socket(self, node_state):
        chunk_size = node_state.chunk_size
        weights_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        weights_server.bind(("0.0.0.0", 5002))
        weights_server.listen(1)
        weights_cli = weights_server.accept()[0]
        weights_cli.setblocking(0)
        
        model_weights = self._recv_weights(weights_cli, chunk_size)
        node_state.weights = model_weights
        weights_server.close()

    def _recv_weights(self, sock: socket.socket, chunk_size: int):
        size_left = 8
        byts = bytearray()
        while size_left > 0:
            try: 
                recv = sock.recv(min(size_left, 8))
                size_left -= len(recv)
                byts.extend(recv)
            except socket.error as e:
                if e.errno != socket.EAGAIN:
                    raise e
                select.select([sock], [], [])
        array_len = int.from_bytes(byts, 'big')
        
        weights = []
        for i in range(array_len):
            recv = bytes(socket_recv(sock, chunk_size))
            weights.append(self._decomp(recv))
        return weights
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        return zfpy.decompress_numpy(lz4.frame.decompress(byts))
    def _data_server(self, node_state: NodeState, to_send: Queue):
        chunk_size = node_state.chunk_size
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_server.bind(("0.0.0.0", 5000))
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setblocking(0)

        while True:
            data = bytes(socket_recv(data_cli, chunk_size))
            inpt = zfpy.decompress_numpy(data)
            to_send.put(inpt)

    def _data_client(self, node_state: NodeState, to_send: Queue):
        graph = tf.get_default_graph()
        while node_state.next_node == "":
            time.sleep(5)# Wait until next_node is set by model socket
        chunk_size = node_state.chunk_size
        model = node_state.model
        next_node_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        next_node_client.connect((node_state.next_node, 5000))
        next_node_client.setblocking(0)
        
        while True:
            inpt = to_send.get()
            with graph.as_default():
                output = model.predict(inpt)
            out = self._comp(output)
            socket_send(out, next_node_client, chunk_size)

    def run(self):
        ns = NodeState(chunk_size = 512 * 1000)
        m = Thread(target=self._model_socket, args=(ns,))
        w = Thread(target=self._weights_socket, args=(ns,))
        to_send = queue.Queue(1000) # Arbitrary size of queue, can change later
        dserv = Thread(target=self._data_server, args=(ns, to_send))
        dcli = Thread(target=self._data_client, args=(ns, to_send))
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

