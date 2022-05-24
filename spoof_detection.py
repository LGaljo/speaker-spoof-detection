import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from math import ceil, floor

from keras import layers
from keras import models

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

# Set the seed value for experiment reproducibility.
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)
from dataset_preprocessing import preprocess_dataset, fetch_dataset
from test_spoof_detection_model import predict_file, run_test

labels = ['spoof', 'genuine']

AUTOTUNE = tf.data.AUTOTUNE

DATASET_PATH_TRAIN = 'DS_10283_3055/ASVspoof2017_V2_train'
DATASET_PATH_DEV = 'DS_10283_3055/ASVspoof2017_V2_dev'
DATASET_PATH_EVAL = 'DS_10283_3055/ASVspoof2017_V2_eval'

DATASET_PATH_TRAIN_LABELS = 'DS_10283_3055/protocol_V2/ASVspoof2017_V2_train.trn.txt'
DATASET_PATH_EVAL_LABELS = 'DS_10283_3055/protocol_V2/ASVspoof2017_V2_eval.trl.txt'
DATASET_PATH_DEV_LABELS = 'DS_10283_3055/protocol_V2/ASVspoof2017_V2_dev.trl.txt'


def load_model():
    new_model = tf.keras.models.load_model('saved_model/model')
    new_model.summary()
    return new_model


def create_model():
    # Create CNN model
    model = models.Sequential([
        layers.Input(shape=(61, 513, 1)),
        # Downsample the input.
        layers.Resizing(64, 64),
        # Normalize.
        # norm_layer,
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(labels), activation='softmax'),
    ])

    # Tell more about model
    model.summary()

    # Prepare for training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def train_model(dataset):
    train_size = ceil(num_samples * 0.9)
    val_size = num_samples - train_size

    dataset = dataset.shuffle(128)
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(val_size)
    val_ds = val_ds.take(val_size)

    print('Validation set size', val_size)
    # print('Test set size', len(test_files))
    print('Training set size', train_size)

    input_shape = train_ds.take(1).get_single_element(0)[0].shape
    print('Input shape:', input_shape)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))

    # Create a batches with 64 samples
    batch_size = 128
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    print('Input shape: ', train_ds.take(1).get_single_element(0)[0].shape)

    # Set TF to cache files to prevent reading them on each epoch
    # Set TF to prefetch files for next epoch, while training on current batch
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Fit the model to train data
    EPOCHS = 100
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
    )

    # Save model to disk
    model.save('saved_model/model')

    #
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Share')
    plt.show()


if __name__ == '__main__':
    filenames_train, num_samples = fetch_dataset()
    # model = load_model()
    model = create_model()

    # Preprocess dataset
    full_dataset = preprocess_dataset(filenames_train)

    for i in range(1):
        train_model(full_dataset)

    run_test()
