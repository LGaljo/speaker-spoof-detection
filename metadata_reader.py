from tflite_support.metadata_writers import audio_classifier
from tflite_support.metadata_writers import metadata_info
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

# _MODEL_PATH = "tflite_model/yamnet.tflite"
_MODEL_PATH = "tflite_model/spoof_detection_metadata.tflite"
# Task Library expects label files that are in the same format as the one below.


def read_metadata():
    from tflite_support import metadata

    displayer = metadata.MetadataDisplayer.with_model_file(_MODEL_PATH)
    print("Metadata populated:")
    print(displayer.get_metadata_json())

    print("Associated file(s) populated:")
    for file_name in displayer.get_packed_associated_file_list():
        print("file name: ", file_name)
        print("file content:")
        print(displayer.get_associated_file_buffer(file_name))


if __name__ == '__main__':
    read_metadata()
