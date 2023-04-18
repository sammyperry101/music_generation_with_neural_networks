import collections
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import os
import mido
import torch
import torch.nn as nn
from utils import *
import pretty_midi
import RNN_Model
import Run_Transformer_Model
import Transformer_Pitch
import Timings_Dataset
import Timings_Transformer

# This method converts a midi file into an array of pitches and timings to load into
# the Transformer model
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

    # Create arrays to store the note arrays
    note_diffs = getNoteLengths(getMidiTempo(midi))
    note_lengths = getNoteLengths(getMidiTempo(midi))
    note_diffs.insert(0, 0)
    note_lengths = np.asarray(note_lengths)
    note_diffs = np.asarray(note_diffs)

    # Iterate through sorted notes and create information for each note
    for note in sorted_notes:
        start = note.start
        end = note.end
        pitches["pitch"].append(note.pitch)

        # Calculate the diff and length values, then use the arrays containing
        # the different values by note to round to nearest and get index
        diff_value = start - prev_start
        length_value = end - start
        diff_index = (np.abs(note_diffs - diff_value)).argmin()
        length_index = (np.abs(note_lengths - length_value)).argmin()
        diffs["diff"].append(diff_index)
        lengths["length"].append(length_index)
        prev_start = start

    return (
        pd.DataFrame({name: np.array(value) for name, value in pitches.items()}),
        pd.DataFrame({name: np.array(value) for name, value in diffs.items()}),
        pd.DataFrame({name: np.array(value) for name, value in lengths.items()}),
    )


# Checks for tempo information in midi file
# If none, then returns default tempo (120 bpm)
def getMidiTempo(file_name: str):
    midi = mido.MidiFile(file_name)

    # Check for a tempo message in mido file
    for msg in midi:
        if msg.type == "set_tempo":
            return int(round(mido.tempo2bpm(msg.tempo), 0))

    # Return tempo of 120 bpm if no tempo notes found
    return int(round(mido.tempo2bpm(500000), 0))


# Given a tempo, calculate list of all valid note lengths in seconds
# for generated notes
def getNoteLengths(tempo: int):
    # List of valid timings for notes
    timings = [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4,
    ]

    # Calculate length of a crotchet
    crotchet = round(60 / tempo, 2)
    note_lengths = []

    # Calculate all timings in reference to length of crotchet
    for i in timings:
        note_lengths.append(round(i * crotchet, 2))

    return note_lengths


# Parse selected number of files, default is 5
def parseFiles(file_names: list, num_files: int = 5):
    all_pitches = []
    all_diffs = []
    all_lengths = []
    # Iterate through files
    for f in file_names[:num_files]:
        # Get pitches, diffs and lengths for a midi file and append these to relevant arrays
        pitches, diffs, lengths = midiToNotesTransformer(f)
        all_pitches.append(pitches)
        all_diffs.append(diffs)
        all_lengths.append(lengths)

    return all_pitches, all_diffs, all_lengths


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

    # First parse only 5 files here but can do more later to increase accuracy of dataset:
    files_to_parse = 5
    all_pitches, all_diff, all_length = parseFiles(file_names, files_to_parse)

    # Print information about the datasets
    print(all_diff)
    print(all_length)

    # Set variables
    batch_size = 32
    embedded_heads = 30
    num_heads = 8
    num_layers = 3
    diff_values = 17
    len_values = 16
    sequence_length = 25
    vocab = 128

    # Set epochs (default is 10)
    epochs = 4

    # Define dataset and sequence length
    # Code for diff_dataset
    # dataset = Timings_Dataset.TimingsDataset(all_diff, sequence_length, files_to_parse, diff_values)
    # Code for length_dataset
    dataset = Timings_Dataset.TimingsDataset(
        all_length, sequence_length, files_to_parse, len_values
    )

    # Build model and define loss function & optimisation method
    # Code for diff_model
    # model = Timings_Transformer.TimingsTransformer(sequence_length, diff_values, num_heads*embedded_heads, num_heads, num_layers, dropout=0.0).to(Run_Transformer_Model.getDevice())
    # Code for length_model
    model = Timings_Transformer.TimingsTransformer(
        sequence_length,
        len_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    loss_function = nn.BCEWithLogitsLoss()
    optimisation_method = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimisation_method = torch.optim.Adam(model.parameters(), lr=0.01)

    # Set model to train
    model.train()

    # Returns trained model, total loss and total accuracy
    model, total_loss, batch_history = Transformer_Pitch.trainModel(
        model, dataset, optimisation_method, loss_function, epochs, batch_size
    )

    # plot model performance
    Transformer_Pitch.plotPerformance(total_loss, batch_history)

    # saves the model (uncode line for different models)
    # Save diff_models
    # torch.save(model.state_dict(), os.path.join(os.sys.path[0],'Transformer_Model_Diff/Transformer_Model_Diff.pth'))
    # Save length_model
    # torch.save(
    #     model.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "Transformer_Model_Length/Transformer_Model_Length.pth"
    #     ),
    # )

    # torch.save(
    #     model.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "Transformer_Model_Diff_Electronic/Transformer_Model_Diff_Electronic.pth"
    #     ),
    # )

    torch.save(
        model.state_dict(),
        os.path.join(
            os.sys.path[0], "Transformer_Model_Length_Electronic/Transformer_Model_Length_Electronic.pth"
        ),
    )

    # torch.save(
    #     model.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "Transformer_Model_Diff_Jazz/Transformer_Model_Diff_Jazz.pth"
    #     ),
    # )

    # torch.save(
    #     model.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "Transformer_Model_Length_Jazz/Transformer_Model_Length_Jazz.pth"
    #     ),
    # )