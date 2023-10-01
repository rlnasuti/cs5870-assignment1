import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import os
from io import StringIO
import sys

os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/usr/lib/cuda'

# Load and preprocess MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = np.array([tf.image.resize(img[..., tf.newaxis], [32, 32]).numpy() for img in train_images])
test_images = np.array([tf.image.resize(img[..., tf.newaxis], [32, 32]).numpy() for img in test_images])

# Convert to 3 channels
# train_images = np.repeat(train_images[..., np.newaxis], 3, -1)
# test_images = np.repeat(test_images[..., np.newaxis], 3, -1)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the VGG16 model
model = Sequential()
model.add(VGG16(include_top=False, weights=None, input_shape=(32, 32, 1)))
model.add(Flatten())
# model.add(Dense(4096, activation="relu"))
# model.add(Dense(4096, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=16, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Capture the model summary as a string
buffer = StringIO()
sys.stdout = buffer
model.summary()
model_summary = buffer.getvalue()
sys.stdout = sys.__stdout__

print(model_summary)
print("Test Accuracy:", test_acc)


# Get the current directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Create the full path for the results file in the same directory as the script
results_file_name = os.path.join(script_directory, os.path.basename(__file__).replace('.py', '.txt'))

with open(results_file_name, 'w') as f:
    f.write(model_summary)
    f.write(f"Test Accuracy: {test_acc}")