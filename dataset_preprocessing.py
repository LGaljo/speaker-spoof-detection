import os
import pathlib

import tensorflow as tf

labels = ['spoof', 'genuine']
AUTOTUNE = tf.data.AUTOTUNE

DATASET_PATH_TRAIN = 'DS_10283_3055/ASVspoof2017_V2_train'
DATASET_PATH_DEV = 'DS_10283_3055/ASVspoof2017_V2_dev'
DATASET_PATH_EVAL = 'DS_10283_3055/ASVspoof2017_V2_eval'


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

    # Concatenate the waveform with `zero_padding`, which ensures all audio clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.

    return equal_length


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == labels)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
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
