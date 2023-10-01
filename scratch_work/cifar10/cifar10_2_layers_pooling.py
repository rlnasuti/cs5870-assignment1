import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
from io import StringIO
import sys

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess data
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

model = Sequential()

# Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

# Max-Pooling Layer
model.add(MaxPooling2D((2, 2)))

# Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

# Max-Pooling Layer
model.add(MaxPooling2D((2, 2)))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))

# Dropout Layer
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=10,
                    batch_size=2048,
                    validation_data=(test_images, test_labels))

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
