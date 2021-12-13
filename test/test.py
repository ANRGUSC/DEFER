from ..src.dispatcher import DEFER
import threading
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import queue
import time

numNodes = 8
computeNodes = ["IPs COMPUTE NODES HERE"]
defer = DEFER(computeNodes)

model = ResNet50(weights='imagenet', include_top=True)
# Depending on the number of compute nodes, use different number of partitions
# "part_at" is where the graph is split, so there should be one less element in this
# list than the number of partitions you want
part_at = ["add_2", "add_4", "add_6", "add_8", "add_10", "add_12", "add_14"]
img_path = 'YOUR IMAGE HERE'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time_min = 5
run = time_min * 60

start = time.time()
def print_result(q):
    res_count = 0
    while (time.time() - start) < run:
        res = q.get()
        res_count += 1
        print(res.shape)
    print(f"{res_count} results in {time_min} min")
    print(f"Throughput: {res_count / run}")
    exit(0)

input_q = queue.Queue(10)
output_q = queue.Queue(10)

a = threading.Thread(target=defer.run_defer, args=(model, part_at, input_q, output_q), daemon=True)
b = threading.Thread(target=print_result, args=(output_q,))
a.start()
b.start()

for i in range(1000):
    # Whatever input you want
    input_q.put(x)

b.join()