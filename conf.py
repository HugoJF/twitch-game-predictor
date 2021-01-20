import os
import numpy as np
import tensorflow as tf
from glob import glob
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Start
test_data_path = "C:\\Users\\hugo_\\Code\\twitch-game-predictor\\images_cleaned"
img_height = 360
img_width = 640
batch_size = 1
num_of_test_samples = 1


def base(path):
    return os.path.basename(path)


class_names = glob("images_cleaned/*")  # Reads all the folders in which images are present
class_names = list(sorted(map(base, class_names)))  # Sorting them
print('Class names')
print(class_names)

test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode=None,
                                                        shuffle=False,
                                                        seed=42)

model = tf.keras.models.load_model('models/trained-2k')

# Confution Matrix and Classification Report
Y_pred = model.predict(validation_generator, num_of_test_samples // batch_size)
y_pred = np.argmax(Y_pred, axis=1)


cor = validation_generator.classes
file = validation_generator.filenames
for i, pred in enumerate(y_pred):
    if pred == cor[i]:
        continue
    name = class_names[cor[i]]
    print('%s: Predited as %s (%s)' % (file[i], class_names[pred], name))

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=class_names))
