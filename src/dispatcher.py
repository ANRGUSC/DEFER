#!pip install flask
#!pip install numpy
#!pip install tensorflow
#!pip install flask-socketio
#!pip install "python-socketio[async_client]"
#!pip install "python-socketio[client]"
#!pip install simple-websocket

import numpy as np
import tensorflow as tf
from tensorflow import keras

from flask import Flask
from flask_socketio import SocketIO, send, emit
import socketio
import asyncio

from typing import List

from requests import get

cli = socketio.Client(logger=True, engineio_logger=True)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
server = SocketIO(app, logger=True)

computeNodes = ["http://127.0.0.1:5001/", "http://127.0.0.1:5002/"]
dispatchIP = get('https://api.ipify.org').text


def partition(model: tf.keras.Model, partit: List[int]) -> tf.keras.Model:
    #To slice properly, add the first and last layers to the partition list
    partitions = [0]
    partitions.extend(partit)
    partitions.append(len(model.layers) - 1)
    parts = []
    for i in range(1, len(partitions)):
        part = model.layers[partitions[i-1]:partitions[i]]
        parts.append(part)
    
    models = []
    for p in range(len(parts)):
        if p == 0:
            inpt = keras.Input(tensor=model.input)
            print(inpt)
        else:
            inpt = keras.Input(tensor=models[p-1].output)
        print(inpt)
        print([layer.output for layer in parts[p]])
        models.append(
            keras.Model(
                inputs=inpt,
                outputs=[layer.output for layer in parts[p]]
            )
        )
        print(models)
    return models

async def dispatchModels(models: List[tf.keras.Model], nodeIPs: List[str]) -> None:
    for i in range(len(models)):
        client = socketio.AsyncClient(logger=True, engineio_logger=True)
        model_json = models[i].to_json()
        print(nodeIPs[i])
        await client.connect(nodeIPs[i], namespaces=['/recv_model'], auth={"name": "dispatcher"})
        if i != len(models) - 1:
            nextNode = nodeIPs[i + 1]
        else:
            # Reached the end of the nodes, the last node needs to point back to the dispatcher
            nextNode = dispatchIP
        
        print("Reached emit")
        await client.emit("dispatch", data=(model_json, nextNode), namespace='/recv_model')
    await client.sleep(2)

def startDistEdgeInference(client, model_input: tf.Tensor, send_to: str):
    print("Starting inference")
    client.connect(send_to, namespaces=['/recv_data'])
    client.emit("data", data=model_input.numpy().tolist(), namespace='/recv_data')


#models_to_dispatch = partition(model, [1])


@server.on('data', namespace="/recv_data")
def got_result(data):
    print("Done distributing, result is {}".format(data))

@server.on('connect', namespace="/recv_data")
def data_connect():
    print("Previous node connected")
    startDistEdgeInference(cli, inpt, computeNodes[0])


async def main():
    await dispatchModels(models_to_dispatch, computeNodes)
    #await startDistEdgeInference(sio, inpt, computeNodes[0])


if __name__ == '__main__':
    asyncio.run(main())
    server.run(app, debug=True, port=5000, use_reloader=False)


