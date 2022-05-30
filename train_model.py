from math import ceil

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Sequential

# Set the seed value for experiment reproducibility.
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)
from dataset_preprocessing import preprocess_dataset, fetch_dataset
from preprocess_layer import PreprocessTFLayer
from test_model import run_test

labels = ['spoof', 'genuine']

AUTOTUNE = tf.data.AUTOTUNE

DATASET_PATH_DEV = 'DS_10283_3055/ASVspoof2017_V2_dev'


def load_model():
    new_model = tf.keras.models.load_model('saved_model/model')
    new_model.summary()
    return new_model


def build_model(sample_rate=16000, duration=1):
    n_samples = sample_rate * duration

    _model = Sequential(name="spoof_detection")
    _model.add(layers.Input(shape=(n_samples,), name="waveform_input", dtype='float32'))
    # _model.add(DebugLayer())
    _model.add(PreprocessTFLayer())
    _model.add(layers.BatchNormalization(axis=2))
    _model.add(layers.Resizing(64, 64))
    _model.add(layers.Conv2D(64, 3, activation='relu'))
    _model.add(layers.Conv2D(128, 3, activation='relu'))
    _model.add(layers.MaxPooling2D())
    _model.add(layers.Dropout(0.25))
    _model.add(layers.Flatten())
    _model.add(layers.Dense(256, activation='relu'))
    _model.add(layers.Dropout(0.5))
    _model.add(layers.Dense(len(labels), activation='softmax', name="probabilities"))

    return _model


def create_model():
    # Create CNN model

    model = build_model()

    # Tell more about model
    model.summary()

    # Prepare for training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    return model


def train_model(dataset):
    train_size = ceil(num_samples * 0.9)
    val_size = num_samples - train_size

    dataset = dataset.shuffle(128)
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(val_size)
    val_ds = val_ds.take(val_size)

    print('Validation set size', val_size)
    # print('Test set size', len(test_files))
    print('Training set size', train_size)

    input_shape = train_ds.take(1).get_single_element(0)[0].shape
    print('Input shape:', input_shape)

    # Create a batches with 64 samples
    batch_size = 128
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Set TF to cache files to prevent reading them on each epoch
    # Set TF to prefetch files for next epoch, while training on current batch
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Fit the model to train data
    EPOCHS = 100
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
    )

    # Save model to disk
    model.save('saved_model/model')

    # Show training progress
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    filenames_train, num_samples = fetch_dataset()
    # model = load_model()
    model = create_model()

    # Preprocess dataset
    full_dataset = preprocess_dataset(filenames_train)

    for i in range(1):
        train_model(full_dataset)

    run_test(model, DATASET_PATH_DEV)
