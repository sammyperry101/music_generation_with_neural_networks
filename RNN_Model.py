import collections
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import tensorflow as tf
from matplotlib import pyplot as plt
import os

# Download the selected files to use
def downloadFiles(directory: str):
    if not directory.exists():
        tf.keras.utils.get_file(
            "maestro-v3.0.0-midi.zip",
            origin="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
            cache_dir=".",
            extract=True,
            cache_subdir="data",
        )


# This method converts a midi file into an array of notes to load into
# machine learning algorithm
def midiToNotes(midi: str):
    pm = pretty_midi.PrettyMIDI(midi)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    # Iterate through sorted notes and create information for each note
    for note in sorted_notes:
        start = note.start
        end = note.end
        notes["pitch"].append(note.pitch)
        notes["diff"].append(start - prev_start)
        notes["length"].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


# Normalize note pitch
def scalePitch(pitches: tf.data.Dataset):
    pitches = pitches / [vocab, 1.0, 1.0]
    return pitches


# Split the labels
def splitLabels(sequences: tf.data.Dataset):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

    return scalePitch(inputs), labels


# This code is used to create sequences used to train model on
# sequence_length is length of sequences so use different numbers to test this
def createSequences(dataset: tf.data.Dataset, sequence_length: int):

    sequence_length += 1

    # Take 1 extra for the labels
    windows = dataset.window(sequence_length, shift=1, stride=1, drop_remainder=True)

    # flat_map flattens the dataset of datasets into a dataset of tensors
    flatten = lambda window: window.batch(sequence_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    return sequences.map(splitLabels, num_parallel_calls=tf.data.AUTOTUNE)


# This function applies positive pressure to make sure values are positive
def meanSquareErr(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    pos_pressure = 5 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + pos_pressure)


# Parse selected number of files, default is 5
def parseFiles(file_names: list, num_files: int = 5):
    all_notes = []
    for f in file_names[:num_files]:
        notes = midiToNotes(f)
        all_notes.append(notes)

    return pd.concat(all_notes)


# Train the model based on selected number of epochs
def trainModel(model: tf.keras.Model, epochs: int = 10):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./training_checkpoints/ckpt_{epoch}", save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=10, verbose=1, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    return model, history


# Create plot to show performance by epoch and change over time
def plotPerformance(model_history: tf.keras.callbacks.History):
    plt.plot(model_history.epoch, model_history.history["loss"], label="total loss")
    plt.show()


if __name__ == "__main__":
    # Random seed set, can be removed if needed but then will be different every time
    seed = 55
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Train on classical dataset
    # Here initialise where to download data from and if not exist then downloads
    # data_dir = pathlib.Path(os.path.join(os.sys.path[0], "data/maestro-v3.0.0"))
    # downloadFiles(data_dir)

    # Get list of filenames, used to train data
    # file_names = glob.glob(str(data_dir / "**/*.mid*"))

    # Train on electronic dataset
    data_dir = pathlib.Path(os.path.join(os.sys.path[0], "data/Electronic"))

    # Train on Jazz dataset
    # data_dir = pathlib.Path(os.path.join(os.sys.path[0], "data/Jazz"))

    # Get list of filenames, used to train data
    file_names = glob.glob(str(data_dir / "*.mid"))

    # First parse only 5 files here but can do more later to increase accuracy of dataset:
    files_to_parse = 5
    all_notes = parseFiles(file_names, files_to_parse)

    # Next create tensorflow dataset from this info:
    key_order = ["pitch", "diff", "length"]
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    notes_dataset = tf.data.Dataset.from_tensor_slices(train_notes)

    # Create the sequences here:
    # Define length of sequences
    sequence_length = 100  
    # Define vocab (number available pitches) - default is 128
    vocab = 128  
    sequence_dataset = createSequences(notes_dataset, sequence_length)

    # Set batch size
    batch_size = 64
    # Buffer size is total number of sequences in dataset
    buffer_size = len(all_notes) - sequence_length
    # Define training dataset
    train_ds = (
        sequence_dataset.shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Has three outputs, one for each variable
    # For diff and length, encourage model to produce non-negative results using mean square err function above
    input_shape = (sequence_length, 3)
    # Define a learning rate
    learning_rate = 0.01

    # Define input layers
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(vocab)(inputs)

    # Define output layers
    outputs = {
        "pitch": tf.keras.layers.Dense(128, name="pitch")(x),
        "diff": tf.keras.layers.Dense(1, name="diff")(x),
        "length": tf.keras.layers.Dense(1, name="length")(x),
    }

    # Define model structure
    model = tf.keras.Model(inputs, outputs)

    # Define loss metrics for each metric
    loss = {
        "pitch": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "diff": meanSquareErr,
        "length": meanSquareErr,
    }

    # Define optimiser
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile giving less weight to pitch due to higher loss value otherwise
    model.compile(
        loss=loss,
        loss_weights={
            "pitch": 0.01,
            "diff": 1.0,
            "length": 1.0,
        },
        optimizer=optimiser,
    )

    # Train and return losses of model
    model_losses = model.evaluate(train_ds, return_dict=True)

    # Now train the model
    model, model_history = trainModel(model, epochs=10)

    # Plot performance of the model
    plotPerformance(model_history)

    # Save the Model
    # model.save(os.path.join(os.sys.path[0], "RNN-Model"))
    model.save(os.path.join(os.sys.path[0], "RNN-Electronic"))
    # model.save(os.path.join(os.sys.path[0], "RNN-Jazz"))