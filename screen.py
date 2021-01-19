from screengrab import grab_screen
from tensorflow import keras
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import time
import os

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
    screen = grab_screen(region=(-1920, 128, -340, 1020))
    # Resize to input into CNN
    screen = cv2.resize(screen, (w, h))

    # Do something
    img_array = keras.preprocessing.image.img_to_array(screen)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predict
    s = time.time()
    prediction = model.predict(img_array)
    e = time.time()

    # Compute confidence
    score = tf.nn.softmax(prediction[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence ({:.2f} ms)."
            .format(class_names[np.argmax(score)], 100 * np.max(score), ((e - s) * 1000))
    )

    # Update preview
    if preview:
        plt.imshow(screen)
        plt.title("asd")
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)

    time.sleep(interval)
