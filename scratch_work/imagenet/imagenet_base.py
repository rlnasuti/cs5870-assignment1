import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
from io import StringIO
import sys
import xml.etree.ElementTree as ET

def create_label_mapping(root_dir):
    train_label_dir = os.path.join(root_dir, 'imagenet', 'dataset', 'ILSVRC', 'Annotations', 'CLS-LOC', 'train')
    unique_labels = set()

    for label_subdir in os.listdir(train_label_dir): 
        subdir_path = os.path.join(train_label_dir, label_subdir)
        for label_file in os.listdir(subdir_path):  # Iterate through files in each sub-directory
            label_path = os.path.join(subdir_path, label_file)
            label = parse_xml_annotation(label_path)
            unique_labels.add(label)

    return {label: idx for idx, label in enumerate(unique_labels)}

def parse_xml_annotation(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    object_elem = root.find('object')
    name_elem = object_elem.find('name')
    return name_elem.text

def load_and_preprocess_image(image_path, label_path):
    def _load_and_process(img_path, lbl_path):
        # Convert tensors to numpy arrays and then to strings
        img_path, lbl_path = img_path.numpy().decode(), lbl_path.numpy().decode()

        # Image processing
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image /= 255.0

        # Label processing
        label = parse_xml_annotation(lbl_path)
        label_idx = label_mapping[label]
        label_one_hot = tf.one_hot(label_idx, len(label_mapping))

        return image, label_one_hot
    
    image, label = tf.py_function(_load_and_process, [image_path, label_path], [tf.float32, tf.float32])
    image.set_shape([224, 224, 3])
    label.set_shape([len(label_mapping)])
    return image, label



def build_dataset(image_paths, label_paths):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def get_val_image_and_label_paths(image_dir, label_dir):
    image_paths = []
    label_paths = []

    # Get all image files from the directory
    img_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    # Ensure every image has a label
    for img_file in img_files:
        corresponding_label = img_file.replace(".JPEG", ".xml").replace(".jpeg", ".xml")

        if corresponding_label in label_files:
            image_paths.append(os.path.join(image_dir, img_file))
            label_paths.append(os.path.join(label_dir, corresponding_label))
        else:
            print(f"Warning: Missing label for {img_file}")

    return image_paths, label_paths


def get_image_and_label_paths(image_dir, label_dir):
    image_paths = []
    label_paths = []

    subdirs = sorted(os.listdir(image_dir))
    for subdir in subdirs:
        img_subdir_path = os.path.join(image_dir, subdir)
        label_subdir_path = os.path.join(label_dir, subdir)

        # Ensure the label directory exists
        if not os.path.exists(label_subdir_path):
            print(f"Warning: Missing label directory for {subdir}")
            continue

        img_files = sorted(os.listdir(img_subdir_path))
        label_files = sorted(os.listdir(label_subdir_path))

        # Ensure every image has a label
        for img_file in img_files:
            corresponding_label = img_file.replace(".JPEG", ".xml").replace(".jpeg", ".xml")

            if corresponding_label in label_files:
                image_paths.append(os.path.join(img_subdir_path, img_file))
                label_paths.append(os.path.join(label_subdir_path, corresponding_label))
            else:
                print(f"Warning: Missing label for {img_file}")

    return image_paths, label_paths



def create_imagenet_datasets(root_dir):
    train_image_dir = os.path.join(root_dir, 'imagenet', 'dataset', 'ILSVRC', 'Data', 'CLS-LOC', 'train')
    train_label_dir = os.path.join(root_dir, 'imagenet', 'dataset', 'ILSVRC', 'Annotations', 'CLS-LOC', 'train')

    val_image_dir = os.path.join(root_dir, 'imagenet', 'dataset', 'ILSVRC', 'Data', 'CLS-LOC', 'val')
    val_label_dir = os.path.join(root_dir, 'imagenet', 'dataset', 'ILSVRC', 'Annotations', 'CLS-LOC', 'val')

    train_image_paths, train_label_paths = get_image_and_label_paths(train_image_dir, train_label_dir)
    val_image_paths, val_label_paths = get_val_image_and_label_paths(val_image_dir, val_label_dir)

    train_dataset = build_dataset(train_image_paths, train_label_paths)
    val_dataset = build_dataset(val_image_paths, val_label_paths)

    return train_dataset, val_dataset, train_image_paths, val_image_paths

global label_mapping
label_mapping = create_label_mapping('./')
train_dataset, val_dataset, train_image_paths, val_image_paths = create_imagenet_datasets('./')

# Adjust dataset for training: shuffle, batch, and repeat
train_dataset = train_dataset.shuffle(10000).batch(32).repeat().prefetch(tf.data.experimental.AUTOTUNE)

# Batch validation dataset
val_dataset = val_dataset.batch(32)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_dataset,
                    epochs=10,
                    steps_per_epoch=len(train_image_paths) // 32,  # Define steps per epoch
                    validation_data=val_dataset,
                    validation_steps=len(val_image_paths) // 32,  # Define validation steps
                    callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(val_dataset, steps=len(val_image_paths) // 32)

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
