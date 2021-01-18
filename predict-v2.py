import numpy as np
import tensorflow as tf

from tensorflow import keras
from time import time
import sys
import os
from glob import glob

img_height = 360
img_width = 640

if len(sys.argv) != 2:
    print('predict-v2.py <image-url>')
    exit(1)


def base(path):
    return os.path.basename(path)

class_names = glob("images_cleaned/*")  # Reads all the folders in which images are present
class_names = list(sorted(map(base, class_names)))  # Sorting them
print('Class names')
print(class_names)

model = tf.keras.models.load_model('models/trained')


sunflower_path = tf.keras.utils.get_file('eu-%f' % time(), origin=sys.argv[1])
img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

for i in range(0, 20):
    print()
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
