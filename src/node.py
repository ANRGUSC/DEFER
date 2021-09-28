import numpy as np
import tensorflow as tf
from tensorflow import keras

import socketio
import asyncio

from flask import Flask
from flask_socketio import SocketIO

sio = socketio.Client()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
server = SocketIO(app, logger=True, engineio_logger=True)

@server.on('data', namespace="/recv_data")
def got_data(data):
    print("Received data in part1")
    inpt = tf.convert_to_tensor(data)
    out = model(inpt).numpy().tolist()
    sio.emit('data', data=out, namespace="/recv_data")

@server.on('connect', namespace="/recv_model")
def connect(auth):
    print("Dispatcher Connected, auth:", auth["name"])

@server.on('disconnect', namespace="/recv_model")
def disconnect():
    print("Dispatcher disconnected")

@server.on('connect', namespace="/recv_data")
def connect():
    print("Previous node connected")

@server.on('dispatch', namespace="/recv_model")
def recv_model(model_json, next_node):
    global model
    print(model_json)
    model = tf.keras.models.model_from_json(model_json)
    #model.save("models/part1")
    print(next_node)
    sio.connect(next_node, namespaces='/recv_data')
    print(sio.connected)

if __name__ == '__main__':
    server.run(app, port=5001, debug=True, use_reloader=False)


