import tensorflow as tf


# from preprocess_tensorflow import preprocess_tf


class PreprocessTFLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_tf", **kwargs):
        super(PreprocessTFLayer, self).__init__(name=name, **kwargs)
        # self.preprocess = preprocess_tf

    def get_config(self):
        config = super(PreprocessTFLayer, self).get_config()
        return config

    def build(self, input_shape):
        # self.non_trainable_weights.append(self.mel_filterbank)
        super(PreprocessTFLayer, self).build(input_shape)

    # def call(self, _input):
    #     print("Input to preprocessing layer", _input.shape)
    #     return self.preprocess(_input)

    def call(self, waveforms):
        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=512,
                                      frame_step=256,
                                      fft_length=1024,
                                      pad_end=True)

        spectrograms = tf.abs(spectrograms)

        spectrograms = tf.expand_dims(spectrograms, 3)

        return spectrograms
