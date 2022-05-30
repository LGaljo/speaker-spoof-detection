import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# Set the seed value for experiment reproducibility.
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)
from dataset_preprocessing import preprocess_dataset, fetch_dataset

labels = ['spoof', 'genuine']

AUTOTUNE = tf.data.AUTOTUNE

DATASET_PATH_TRAIN = 'DS_10283_3055/ASVspoof2017_V2_train'
DATASET_PATH_DEV = 'DS_10283_3055/ASVspoof2017_V2_dev'
DATASET_PATH_EVAL = 'DS_10283_3055/ASVspoof2017_V2_eval'
OWNDATASET_PATH = 'owndataset'


def load_model():
    new_model = tf.keras.models.load_model('saved_model/model')
    new_model.summary()
    return new_model


def run_test(model, path_to_dataset):
    data_dir_dev = pathlib.Path(path_to_dataset)
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

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("Specificity ", TN / (TN + FP))
    print("Sensitivity/Recall ", recall)
    print("Precision ", precision)

    print("F1 Score ", 2 * ((precision * recall) / (precision + recall)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=labels,
                yticklabels=labels,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def predict_file(model, filepath):
    print(filepath)
    sample_ds = preprocess_dataset([str(filepath)])

    # Predict and draw the predictions
    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        # print(prediction)
        if prediction.numpy()[0][0] < prediction.numpy()[0][1]:
            print("Genuine", prediction)
        else:
            print("Spoof", prediction)
        plt.bar(labels, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{filepath}" "{labels[label[0]]}"')
        plt.show()


if __name__ == '__main__':
    filenames_train, num_samples = fetch_dataset()
    _model = load_model()

    # run_test(_model, DATASET_PATH_DEV)
    # run_test(_model, OWNDATASET_PATH)

    for root, dirs, filenames in os.walk('owndataset', topdown=False):
        for filename in filenames:
            predict_file(_model, os.path.join(*[root, filename]))
