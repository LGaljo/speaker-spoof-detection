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


def run_test():
    data_dir_dev = pathlib.Path(DATASET_PATH_DEV)
    filenames_test = tf.io.gfile.glob(str(data_dir_dev) + '/*')
    test_ds = preprocess_dataset(filenames_test)

    print('Number of total test samples:', len(filenames_test))

    #
    test_audio = []
    test_labels = []

    #
    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    #
    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    #
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    #
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    #
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for value in zip(y_true, y_pred):
        if value[0] == 0:
            if value[1] == 0:
                TN += 1
            else:
                FP += 1
        else:
            if value[1] == 0:
                FN += 1
            else:
                TP += 1

    print(TP, TN, FP, FN)
    print("True  positive ", TP)
    print("True  negative ", TN)
    print("False positive ", FP)
    print("False negative ", FN)

    print("Specificity ", TN / (TN + FP))
    print("Sensitivity ", TP / (TP + FN))

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=labels,
                yticklabels=labels,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def predict_file(filepath):
    print(filepath)
    sample_ds = preprocess_dataset([str(filepath)])

    # Predict and draw the predictions
    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        # print(prediction)
        if prediction.numpy()[0][0] < prediction.numpy()[0][1]:
            print("Genuine", prediction)
        else:
            print("Spoof", prediction)
        plt.bar(labels, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{filepath}" "{labels[label[0]]}"')
        plt.show()


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
    model = load_model()

    # run_test()

    predict_file('owndataset/genuine/genuine_Luka_s20fe_1.wav')
    #
    # for root, dirs, filenames in os.walk('owndataset', topdown=False):
    #     for filename in filenames:
    #         predict_file(os.path.join(*[root, filename]))

