import collections
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
from utils import *
import array
import RNN_Model
import Run_Transformer_Model
import Pitches_Dataset
import Pitches_Transformer

# Parse selected number of files, default is 5
def parseFiles(file_names: list, num_files: int = 5):
    all_pitches = []
    all_diffs = []
    all_lengths = []
    # Iterate through files
    for f in file_names[:num_files]:
        # Get the pitches, diffs and lengths and add to arrays of all timings/pitches
        pitches, diffs, lengths = midiToNotesTransformer(f)
        all_pitches.append(pitches)
        all_diffs.append(diffs)
        all_lengths.append(lengths)

    return all_pitches, all_diffs, all_lengths


# This method converts a midi file into an array of pitches and timings to load into the Transformer model
def midiToNotesTransformer(midi: str):
    # Create a PrettyMIDI object to collect info from
    pm = pretty_midi.PrettyMIDI(midi)
    instrument = pm.instruments[0]
    pitches = collections.defaultdict(list)
    diffs = collections.defaultdict(list)
    lengths = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    # Iterate through sorted notes and create information for each note
    for note in sorted_notes:
        start = note.start
        end = note.end
        pitches["pitch"].append(note.pitch)
        diffs["diff"].append(start - prev_start)
        lengths["length"].append(end - start)
        prev_start = start

    return (
        pd.DataFrame({name: np.array(value) for name, value in pitches.items()}),
        pd.DataFrame({name: np.array(value) for name, value in diffs.items()}),
        pd.DataFrame({name: np.array(value) for name, value in lengths.items()}),
    )


# Create the vocabulary for the transformer model pitches
def createVocab(vocab_size: int = 128):
    vocab = []

    # Iterate through pitches, put in one hot encoding for transformer to use
    for pitch in range(0, vocab_size):
        arr = array.array("i", (0 for _ in range(0, vocab_size)))
        arr[pitch] = 1
        arr = list(arr)
        vocab.append(str(arr))

    return list(vocab)


# This method trains the model
def trainModel(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimisation_method: torch.optim.Adam,
    loss_function: nn.BCEWithLogitsLoss,
    epochs: int = 10,
    batch_size: int = 32,
):
    # Create arrays to track total loss and batch history of the model
    total_loss = []
    batch_history = []

    # Run model for defined number of epochs
    for epoch in range(epochs):
        print("Epoch: " + str(epoch))

        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, drop_last=True
        )
        num_trained_batches = 0

        # Iterate through data loaded
        for x, y in data_loader:
            # Format data input
            num_trained_batches += 1
            x = x.to(Run_Transformer_Model.getDevice())
            y = y.float().to(Run_Transformer_Model.getDevice())

            # Format input and expected output
            optimisation_method.zero_grad()
            x = x.swapaxes(1, 0)
            y = y.swapaxes(1, 0)

            # Run model
            output = model(x)

            # Calculate the loss and update model using this
            loss = loss_function(output, y)
            loss.backward()
            optimisation_method.step()

            # Output results while running
            if num_trained_batches % 50 == 0:
                print(f"Batches:  {num_trained_batches}/{len(data_loader)}")
                print(f"Loss: {loss}")
                total_loss.append(loss.item())
                batch_history.append(num_trained_batches)

    return model, total_loss, batch_history


# This method plots the performance of the model
def plotPerformance(total_loss: list, batch_history: list):
    plt.plot(batch_history, total_loss, label="Generator loss")
    plt.show()


if __name__ == "__main__":
    # Random seed set, can be removed if needed but then will be different every time
    seed = 55
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Here initialise where to download data from and if not exist then downloads
    # data_dir = pathlib.Path(os.path.join(os.sys.path[0], "data/maestro-v3.0.0"))
    # RNN_Model.downloadFiles(data_dir)

    # # Get list of filenames, used to train data
    # file_names = glob.glob(str(data_dir / "**/*.mid*"))

    # Train on electronic dataset
    data_dir = pathlib.Path(os.path.join(os.sys.path[0], "data/Electronic"))

    # Train on Jazz dataset
    # data_dir = pathlib.Path(os.path.join(os.sys.path[0], "data/Jazz"))

    # Get list of filenames, used to train data
    file_names = glob.glob(str(data_dir / "*.mid"))

    # First parse only 5 files here but can do more later to increase accuracy of dataset
    files_to_parse = 10
    all_pitches, all_diff, all_length = parseFiles(file_names, files_to_parse)

    # Create the vocabulary (one-hot encoding) for the pitches
    vocab = createVocab()

    # Define dataset and sequence length (default is 25)
    sequence_length = 25
    dataset = Pitches_Dataset.PitchesDataset(
        vocab, all_pitches, sequence_length, files_to_parse
    )

    # Set variables
    batch_size = 32
    embedded_heads = 30
    num_heads = 8
    num_layers = 3
    diff_values = 17
    len_values = 16

    # Set epochs (default is 10)
    epochs = 10

    # Build model and define loss function & optimisation method
    pitch_model = Pitches_Transformer.PitchesTransformer(
        sequence_length,
        len(vocab),
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    loss_function = nn.BCEWithLogitsLoss()
    optimisation_method = torch.optim.Adam(pitch_model.parameters(), lr=0.01)

    # Set model to train
    pitch_model.train()

    # Train the model
    # Returns trained model, total loss and batch_history
    pitch_model, total_loss, batch_history = trainModel(
        pitch_model, dataset, optimisation_method, loss_function, epochs, batch_size
    )

    # Plot model performance
    plotPerformance(total_loss, batch_history)

    # Saves the model
    # torch.save(
    #     pitch_model.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "Transformer_Model_Pitch/Transformer_Model_Pitch.pth"
    #     ),
    # )

    torch.save(
        pitch_model.state_dict(),
        os.path.join(
            os.sys.path[0], "Transformer_Model_Pitch_Electronic/Transformer_Model_Pitch_Electronic.pth"
        ),
    )

    # torch.save(
    #     pitch_model.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "Transformer_Model_Pitch_Jazz/Transformer_Model_Pitch_Jazz.pth"
    #     ),
    # )