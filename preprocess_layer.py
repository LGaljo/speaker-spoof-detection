import tensorflow as tf
from preprocess_tensorflow import preprocess_tf


class PreprocessTFLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_tf", **kwargs):
        super(PreprocessTFLayer, self).__init__(name=name, **kwargs)
        self.preprocess = preprocess_tf

    def call(self, _input):
        return self.preprocess(_input)

    def get_config(self):
        config = super(PreprocessTFLayer, self).get_config()
        return config
