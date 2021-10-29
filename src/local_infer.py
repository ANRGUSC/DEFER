from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing import image
import numpy as np
import time

model = ResNet50(weights='imagenet', include_top=True)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time_run = 10
in_sec = time_run * 60
start = time.time()
result_count = 0
while (time.time() - start) < in_sec:
    model.predict(x)
    result_count += 1
print(f"In {time_run} min, {result_count} results")