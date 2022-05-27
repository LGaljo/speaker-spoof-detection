import tensorflow as tf
import numpy as np


@tf.function
def preprocess_tf(x):

    print(x)
    print(x[0])

    input_len = 15600
    waveform = x[:input_len]

    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)

    # # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)

    print("Shape", waveform.shape)
    print("Shape", zero_padding.shape)

    # # Concatenate the waveform with `zero_padding`, which ensures all audio clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 1)
    # # Convert the waveform to a spectrogram via a STFT.

    # Create spectrogram by utilising STFT
    spectrogram = tf.signal.stft(equal_length, frame_length=512, frame_step=256, fft_length=1024)

    # Obtain the magnitude of the STFT.
    # Absolute value of real and imaginary part of fft
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used as image-like input data with
    # convolution layers (which expect shape (`batch_size`, `height`, `width`, `channels`)).
    print("Output of preprocessing layer", spectrogram)

    spectrogram = spectrogram[..., tf.newaxis]
    spectrogram = tf.reshape(spectrogram, shape=(1, 61, 513, 1))

    print("Output of preprocessing layer", spectrogram)

    return spectrogram

