import tensorflow as tf
import numpy as np


@tf.function
def preprocess_tf(x):
    """
    Preprocessing for Keras (MobileNetV2, ResNetV2).
    :param x: np.asarray([image, image, ...], dtype="float32") in RGB
    :return: normalized image tf style (RGB)
    """

    print(x)
    print(x[0])

    input_len = 16000
    x = x[:input_len]
    # zero_padding = tf.zeros([input_len] - tf.shape(x), dtype=tf.float32)
    # x = tf.cast(x, tf.float32)
    #
    # equal_length = tf.concat([x, zero_padding], 0)

    # Create spectrogram by utilising STFT
    spectrogram = tf.signal.stft(x, frame_length=512, frame_step=256, fft_length=1024)

    # Obtain the magnitude of the STFT.
    # Absolute value of real and imaginary part of fft
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used as image-like input data with
    # convolution layers (which expect shape (`batch_size`, `height`, `width`, `channels`)).
    spectrogram = spectrogram[..., tf.newaxis]
    spectrogram = tf.reshape(spectrogram, shape=(1, 61, 513, 1))

    return spectrogram

