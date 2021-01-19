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

modelv1 = tf.keras.models.load_model('models/trained-1k')
modelv2 = tf.keras.models.load_model('models/trained-2k')

sunflower_path = tf.keras.utils.get_file('eu-%f' % time(), origin=sys.argv[1])
img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

s1 = time()
predictionsv1 = modelv1.predict(img_array)
e1 = time()
print("v1 predicted in %d" % ((e1 - s1) * 1000))
scorev1 = tf.nn.softmax(predictionsv1[0])


s2 = time()
predictionsv2 = modelv2.predict(img_array)
e2 = time()
print("v2 predicted in %d" % ((e2 - s2) * 1000))
scorev2 = tf.nn.softmax(predictionsv2[0])

for i in range(0, 20):
    print()
print(
    "v1 = This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(scorev1)], 100 * np.max(scorev1))
)
print(
    "v2 = This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(scorev2)], 100 * np.max(scorev2))
)
