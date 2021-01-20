from screengrab import grab_screen
from tensorflow import keras
from glob import glob
from time import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import time as t
import os
import sys

expected_class = None

if len(sys.argv) != 2:
    print('Running is prediction mode only (not saving unknown images)')
    print('To run in saving mode: screen.py <class_name>')

if len(sys.argv) == 2:
    expected_class = sys.argv[1]

last_save = time()

# CNN input size
w = 640
h = 360

# Amount to sleep between grabs
interval = 0.05

# Use plt to preview input
preview = False

# Create window
if preview:
    plt.figure(figsize=(10, 10))
    plt.show(block=False)


def base(path):
    return os.path.basename(path)


# Hardcoded from model
class_names = ['apex', 'csgo', 'dbd', 'eft', 'fifa-21', 'fortnite', 'gtav', 'lol', 'minecraft', 'poe', 'rocket-league',
               'rust', 'valorant', 'warzone', 'wow']

# Load the bitch
model = tf.keras.models.load_model('models/trained-2k')

while True:
    # Use (0, 0, 1920, 1080) for center/main monitor (if 1080p)
    screen_raw = grab_screen(region=(-1920, 127, -340, 1016))
    # Resize to input into CNN
    screen = cv2.resize(screen_raw, (w, h))

    # Do something
    img_array = keras.preprocessing.image.img_to_array(screen)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predict
    s = time()
    prediction = model.predict(img_array)
    e = time()

    # Compute confidence
    score = tf.nn.softmax(prediction[0])
    pred = class_names[np.argmax(score)]
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence ({:.2f} ms)."
            .format(pred, 100 * np.max(score), ((e - s) * 1000))
    )

    if expected_class is not None and pred != expected_class and (time() - last_save > 1):
        dir = r'./learning/%s' % expected_class
        path = r'%s/%d.jpg' % (dir, time() * 1000)
        try:
            os.mkdir(dir)
        except:
            pass
        print('Writing image that was predicted wrong to %s' % path)
        cv2.imwrite(path, cv2.cvtColor(screen_raw, cv2.COLOR_BGR2RGB))
        last_save = time()

    # Update preview
    if preview:
        plt.imshow(screen)
        plt.title("asd")
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)

    t.sleep(interval)
