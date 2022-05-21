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
labels = ['spoof', 'genuine']

AUTOTUNE = tf.data.AUTOTUNE

DATASET_PATH_TRAIN = 'DS_10283_3055/ASVspoof2017_V2_train'
DATASET_PATH_DEV = 'DS_10283_3055/ASVspoof2017_V2_dev'
DATASET_PATH_EVAL = 'DS_10283_3055/ASVspoof2017_V2_eval'

DATASET_PATH_TRAIN_LABELS = 'DS_10283_3055/protocol_V2/ASVspoof2017_V2_train.trn.txt'
DATASET_PATH_EVAL_LABELS = 'DS_10283_3055/protocol_V2/ASVspoof2017_V2_eval.trl.txt'
DATASET_PATH_DEV_LABELS = 'DS_10283_3055/protocol_V2/ASVspoof2017_V2_dev.trl.txt'


def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.

    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels` axis from the array.
    return tf.squeeze(audio, axis=-1)


# Return the file from path
def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    nameparts = tf.strings.split(input=parts[-1], sep='_')
    return nameparts[0]


# Get label name from path and convert the audio to waveform
def get_waveform_and_label(file_path):
    # Get label from dict
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(equal_length, frame_length=512, frame_step=256, fft_length=1024)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == labels)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    # print(get_waveform_and_label(files_ds.take(1).get_single_element(0)))
    output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


def fetch_dataset():
    data_dir_train = pathlib.Path(DATASET_PATH_TRAIN)
    data_dir_eval = pathlib.Path(DATASET_PATH_EVAL)

    # Print folder names -> LABELS
    print('Commands:', labels)

    # Print number of samples, number of samples per label and example of parsed tensor
    # Parse all filenames in dir /*/* and shuffle them
    filenames_train = tf.io.gfile.glob(str(data_dir_train) + '/*')
    filenames_eval = tf.io.gfile.glob(str(data_dir_eval) + '/*')
    filenames = tf.random.shuffle(filenames_train + filenames_eval)
    num_samples = len(filenames)

    print('Number of total training + eval samples:', num_samples)
    print('Example file tensor:', filenames[0])

    return filenames, num_samples


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
        layers.Dense(len(labels)),
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

    print("My own spoof")
    # Take the file out of test dataset
    sample_file = pathlib.Path('own_tests/spoof_Recording_3.wav')

    # Preprocess single file
    sample_ds = preprocess_dataset([str(sample_file)])

    # Predict and draw the predictions
    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        print(prediction)
        plt.bar(labels, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{labels[label[0]]}"')
        plt.show()

    print("My own genuine")
    # Take the file out of test dataset
    sample_file = pathlib.Path('own_tests/genuine_Recording_1.wav')

    # Preprocess single file
    sample_ds = preprocess_dataset([str(sample_file)])

    # Predict and draw the predictions
    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        print(prediction)
        plt.bar(labels, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{labels[label[0]]}"')
        plt.show()

    print("Theirs spoof")
    # Take the file out of test dataset
    sample_file = pathlib.Path('DS_10283_3055/ASVspoof2017_V2_dev/spoof_D_1001387.wav')

    # Preprocess single file
    sample_ds = preprocess_dataset([str(sample_file)])

    # Predict and draw the predictions
    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        print(prediction)
        plt.bar(labels, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{labels[label[0]]}"')
        plt.show()

    print("Theirs genuine")
    # Take the file out of test dataset
    sample_file = pathlib.Path('DS_10283_3055/ASVspoof2017_V2_dev/genuine_D_1000028.wav')

    # Preprocess single file
    sample_ds = preprocess_dataset([
        str(
            pathlib.Path('DS_10283_3055/ASVspoof2017_V2_dev/genuine_D_1000028.wav')
        ),
        str(
            pathlib.Path('DS_10283_3055/ASVspoof2017_V2_dev/genuine_D_1000107.wav')
        ),
        str(
            pathlib.Path('DS_10283_3055/ASVspoof2017_V2_dev/genuine_D_1000402.wav')
        )
    ])

    # Predict and draw the predictions
    for spectrogram, label in sample_ds.batch(3):
        prediction = model.predict(spectrogram)
        print(prediction)
        print(np.argmax(prediction, axis=1))
        plt.bar(labels, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{labels[label[0]]}"')
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
    EPOCHS = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
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
