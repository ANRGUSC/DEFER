import threading
import select
import socket
import time

class NodeState:
    def __init__(self, chunk_size) -> None:
        self._chunk_size = chunk_size
        self._next_node = ""
        self._model = ""
        self._weights = ""
        self._lock = threading.Lock()
    @property
    def chunk_size(self):
        with self._lock:
            return self._chunk_size
    @property
    def next_node(self):
        with self._lock:
            return self._next_node
    @next_node.setter
    def next_node(self, nx):
        with self._lock:
            self._next_node = nx
    @property
    def model(self):
        with self._lock:
            return self._model
    @model.setter
    def model(self, m):
        with self._lock:
            self._model = m
    @property
    def weights(self):
        with self._lock:
            return self._weights
    @weights.setter
    def weights(self, w):
        print("Weights set")
        with self._lock:
            self._weights = w
    
def socket_send(bytes, sock: socket.socket, chunk_size: int):
    size = len(bytes)
    size_bytes = size.to_bytes(8, 'big')
    while len(size_bytes) > 0:
        try:
            sent = sock.send(size_bytes)
            size_bytes = size_bytes[sent:]
        except socket.error as e:
                if e.errno != socket.EAGAIN:
                    raise e
                #print(f"Blocking w/ {len(size_bytes)} left in size array")
                select.select([], [sock], [])
    for i in range(0, len(bytes), chunk_size):
        if len(bytes) - i < chunk_size:
            chunk = bytes[i:]
        else:
            chunk = bytes[i:i+chunk_size]
        while len(chunk) > 0:
            try:
                cs = sock.send(chunk)
                #print("Sent chunk w/ size", cs)
                chunk = chunk[cs:]
            except socket.error as e:
                if e.errno != socket.EAGAIN:
                    raise e
                #print(f"Blocking w/ {len(chunk)} left in chunk")
                select.select([], [sock], [])

def socket_recv(sock: socket.socket, chunk_size: int):
    #data_json = b''
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
            #print("Blocking while getting size")
            select.select([sock], [], [])
    data_size = int.from_bytes(byts, 'big')
    #print("Got data, len", data_size)
    data_json = bytearray(data_size)
    data_counter = 0
    left = data_size
    while (left > 0):
        try:
            recv = sock.recv(min(left, chunk_size))
            left -= len(recv)
            data_json[data_counter:data_counter+len(recv)] = recv
            data_counter += len(recv)
            #print("Data left to process: ", left)
        except socket.error as e:
            if e.errno != socket.EAGAIN:
                raise e
            select.select([sock], [], [])
    return data_json