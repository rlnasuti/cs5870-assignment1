from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
import os
from io import StringIO
import sys

os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/usr/lib/cuda'

# Load and preprocess MNIST data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert images to 3 channels and resize
train_images = np.array([tf.image.resize(img, [128, 128]).numpy() for img in train_images])
test_images = np.array([tf.image.resize(img, [128, 128]).numpy() for img in test_images])

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the convolutional base
# for layer in base_model.layers:
#     layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with frozen base
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels), callbacks=[early_stop])

# Optional: Unfreeze some layers of the base model and continue training
# for layer in base_model.layers[-4:]:
#     layer.trainable = True

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

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
