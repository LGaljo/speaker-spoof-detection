import tensorflow as tf

saved_model_dir = 'saved_model/model'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('spoof_detection.tflite', 'wb') as f:
    f.write(tflite_model)
