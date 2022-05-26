import tensorflow as tf


def convert_to_tflite():
    saved_model_dir = 'saved_model/model'

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.allow_custom_ops = True

    tflite_model = converter.convert()

    open('tflite_model/spoof_detection.tflite', 'wb').write(tflite_model)


if __name__ == '__main__':
    convert_to_tflite()
