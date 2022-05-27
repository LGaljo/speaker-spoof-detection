import sys

import tensorflow as tf
# from preprocess_tensorflow import preprocess_tf


class DebugLayer(tf.keras.layers.Layer):
    def __init__(self, name="debug_tf", **kwargs):
        super(DebugLayer, self).__init__(name=name, **kwargs)

    def get_config(self):
        config = super(DebugLayer, self).get_config()
        return config

    def build(self, input_shape):
        super(DebugLayer, self).build(input_shape)

    def call(self, _input):
        print("Debug layer", _input)
        tf.print(_input, output_stream=sys.stdout)
        return _input
