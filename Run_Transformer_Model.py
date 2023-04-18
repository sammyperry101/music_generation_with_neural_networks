import collections
import numpy as np
import pandas as pd
import pretty_midi
from matplotlib import pyplot as plt
import os
import torch
from utils import *
from scipy.special import softmax
import random
import warnings
import Run_RNN_Model
import Pitches_Transformer
import Timings_Transformer
import Transformer_Timings
import Transformer_Pitch

warnings.simplefilter(action="ignore", category=FutureWarning)

repetition = 6

# Create Device for cuda to run:
def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # Get the list of note_diff and note_len values to round to
    note_diffs = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(midi)
    )
    note_lengths = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(midi)
    )
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


# Given a time value, round it to nearest valid length
# given the list of valid times
def roundLengthTimings(file_name: str, time: float):
    note_lengths = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(file_name)
    )
    note_lengths = np.asarray(note_lengths)
    i = (np.abs(note_lengths - time)).argmin()
    return note_lengths[i]


# Given a time value, round it to nearest valid diff
# given the list of valid times
# Difference is that 0 is a valid time for diff, NOT for
# length as cannot have notes of 0 length
def roundDiffTimings(file_name: str, time: float):
    note_lengths = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(file_name)
    )
    note_lengths.insert(0, 0)
    note_lengths = np.asarray(note_lengths)
    i = (np.abs(note_lengths - time)).argmin()
    return note_lengths[i]


# Run the model
def runModel(model: Transformer_Timings, input: np.array, batch_size: int, temp: float):
    # Format input for model
    input = input.to_numpy()
    input = np.concatenate(input, axis=0)
    input = torch.tensor(input).long().to(getDevice()).unsqueeze(0)
    input = input.repeat(batch_size, 1)
    input = input.swapaxes(0, 1)

    # Load input into model
    out = model(input)

    # Create visualisation of the model
    # dot = torchviz.make_dot(out, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    # dot.render(directory="Transformer_Model_Diff", view=True)

    # Get the prediction values from the output
    prediction = out[-1, 0, :]
    prediction = prediction.cpu().detach().numpy()
    prediction = prediction * temp
    prediction = softmax(prediction)

    print("PREDICTION")
    print(prediction)

    return prediction


# Generate notes using model based on number of notes
def generateNotesGroupNumNotes(
    pitch_model,
    diff_model,
    length_model,
    pitch: list,
    diff: list,
    length: list,
    sequence_length: int,
    batch_size: int,
    note_diffs,
    note_lengths,
    note_vocab,
    prev_notes,
    temp: float,
    notes_to_generate: int = 25,
):
    generated_notes = []
    prev_start = 0

    # Create defined number of notes
    with torch.no_grad():
        for _ in range(notes_to_generate):
            # Get final set of values from input
            pitch_input = pitch[-sequence_length:]
            diff_input = diff[-sequence_length:]
            length_input = length[-sequence_length:]

            pitch_prediction = runModel(
                pitch_model, pitch_input, batch_size=batch_size, temp=temp
            )
            diff_prediction = runModel(
                diff_model, diff_input, batch_size=batch_size, temp=temp
            )
            length_prediction = runModel(
                length_model, length_input, batch_size=batch_size, temp=temp
            )

            # Using prediction array, select diff from list
            diff_values = np.random.choice(
                note_diffs, size=len(note_diffs), replace=False, p=diff_prediction
            )
            index = random.randint(0, temp)
            new_diff = diff_values[index]
            new_row_diff = {"diff": new_diff}
            diff = diff.append(new_row_diff, ignore_index=True)
            print(new_diff)

            # Using prediction array, select length from list
            length_values = np.random.choice(
                note_lengths, size=len(note_lengths), replace=False, p=length_prediction
            )
            index = random.randint(0, temp)
            new_length = length_values[index]
            new_row_length = {"length": new_length}
            length = length.append(new_row_length, ignore_index=True)
            print(new_length)

            # Using the prediction array, select notes from the vocab
            one_hot_notes = np.random.choice(
                note_vocab, size=sequence_length, replace=False, p=pitch_prediction
            )
            index = 0

            # Get note value as int
            one_hot_note = one_hot_notes[index]
            new_pitch = note_vocab.index(one_hot_note)

            # Iterate through previous selected notes to avoid repetition
            # Select new notes when repetition occurs
            for prev_note in prev_notes:
                while new_pitch == prev_note:
                    index += 1
                    one_hot_note = one_hot_notes[index]
                    new_pitch = note_vocab.index(one_hot_note)

            # Add new note to array
            print(new_pitch)
            new_row = {"pitch": new_pitch}
            pitch = pitch.append(new_row, ignore_index=True)

            # Add new note to previously selected and remove oldest if max repetition size is reached
            if len(prev_notes) == repetition:
                prev_notes.remove(prev_notes[0])

            prev_notes.append(new_pitch)

            # Add generated note to generated_notes list
            start = prev_start + new_diff
            end = start + new_length
            input_note = (new_pitch, new_diff, new_length)
            generated_notes.append((*input_note, start, end))
            prev_start = start

    return generated_notes


# Generate notes based on the number of seconds to generate
def generateNotesGroupNumSeconds(
    pitch_model,
    diff_model,
    length_model,
    pitch: list,
    diff: list,
    length: list,
    sequence_length: int,
    batch_size: int,
    note_diffs,
    note_lengths,
    note_vocab,
    prev_notes,
    temp: float,
    num_seconds: int = 10,
):
    generated_notes = []
    prev_start = 0
    end = 0

    # Create defined number of notes
    with torch.no_grad():
        while end < num_seconds:
            # Get final set of values from input
            pitch_input = pitch[-sequence_length:]
            diff_input = diff[-sequence_length:]
            length_input = length[-sequence_length:]

            pitch_prediction = runModel(
                pitch_model, pitch_input, batch_size=batch_size, temp=temp
            )
            diff_prediction = runModel(
                diff_model, diff_input, batch_size=batch_size, temp=temp
            )
            length_prediction = runModel(
                length_model, length_input, batch_size=batch_size, temp=temp
            )

            # Using prediction array, select diff from list
            diff_values = np.random.choice(
                note_diffs, size=len(note_diffs), replace=False, p=diff_prediction
            )
            index = random.randint(0, temp)
            new_diff = diff_values[index]
            new_row_diff = {"diff": new_diff}
            diff = diff.append(new_row_diff, ignore_index=True)
            print(new_diff)

            # Using prediction array, select length from list
            length_values = np.random.choice(
                note_lengths, size=len(note_lengths), replace=False, p=length_prediction
            )
            index = random.randint(0, temp)
            new_length = length_values[index]
            new_row_length = {"length": new_length}
            length = length.append(new_row_length, ignore_index=True)
            print(new_length)

            # Using the prediction array, select notes from the vocab
            one_hot_notes = np.random.choice(
                note_vocab, size=sequence_length, replace=False, p=pitch_prediction
            )
            index = 0

            # Get note value as int
            one_hot_note = one_hot_notes[index]
            new_pitch = note_vocab.index(one_hot_note)

            # Iterate through previous selected notes to avoid repetition
            # Select new notes when repetition occurs
            for prev_note in prev_notes:
                while new_pitch == prev_note:
                    index += 1
                    one_hot_note = one_hot_notes[index]
                    new_pitch = note_vocab.index(one_hot_note)

            # Add new note to array
            print(new_pitch)
            new_row = {"pitch": new_pitch}
            pitch = pitch.append(new_row, ignore_index=True)

            # Add new note to previously selected and remove oldest if max repetition size is reached
            if len(prev_notes) == repetition:
                prev_notes.remove(prev_notes[0])

            prev_notes.append(new_pitch)

            # Add generated note to generated_notes list
            start = prev_start + new_diff
            end = start + new_length
            input_note = (new_pitch, new_diff, new_length)
            generated_notes.append((*input_note, start, end))
            prev_start = start

    return generated_notes


# Generate notes based on the number of bars to generate
def generateNotesGroupNumBars(
    pitch_model,
    diff_model,
    length_model,
    pitch: list,
    diff: list,
    length: list,
    sequence_length: int,
    batch_size: int,
    note_diffs,
    note_lengths,
    note_vocab,
    prev_notes,
    temp: float,
    num_bars: int = 8,
    file_name: str = "output.mid",
):
    return generateNotesGroupNumSeconds(
        pitch_model,
        diff_model,
        length_model,
        pitch,
        diff,
        length,
        sequence_length=sequence_length,
        batch_size=batch_size,
        note_diffs=note_diffs,
        note_lengths=note_lengths,
        note_vocab=note_vocab,
        prev_notes=prev_notes,
        temp=temp,
        num_seconds=(num_bars / (Transformer_Timings.getMidiTempo(file_name) / 4)) * 60,
    )


def generateFromUI(
    file_name: str, num_to_generate: int, type_to_generate: str, temp: float, genre : str
):
    print("GENERATE TRANSFORMER")

    if genre == "Classical":
        pitch_path = "Transformer_Model_Pitch/Transformer_Model_Pitch.pth"
        diff_path = "Transformer_Model_Diff/Transformer_Model_Diff.pth"
        len_path = "Transformer_Model_Length/Transformer_Model_Length.pth"
    elif genre == "Dance/Electronic":
        pitch_path = "Transformer_Model_Pitch_Electronic/Transformer_Model_Pitch_Electronic.pth"
        diff_path = "Transformer_Model_Diff_Electronic/Transformer_Model_Diff_Electronic.pth"
        len_path = "Transformer_Model_Length_Electronic/Transformer_Model_Length_Electronic.pth"
    elif genre == "Jazz":
        pitch_path = "Transformer_Model_Pitch_Jazz/Transformer_Model_Pitch_Jazz.pth"
        diff_path = "Transformer_Model_Diff_Jazz/Transformer_Model_Diff_Jazz.pth"
        len_path = "Transformer_Model_Length_Jazz/Transformer_Model_Length_Jazz.pth"


    # Set variables
    batch_size = 32
    embedded_heads = 30
    num_heads = 8
    num_layers = 3
    diff_values = 17
    len_values = 16
    sequence_length = 25
    vocab = 128

    # Set variables to be used in generating notes
    prev_notes = []

    # loads the models
    pitch_model = Pitches_Transformer.PitchesTransformer(
        sequence_length,
        vocab,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(getDevice())
    pitch_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), pitch_path
            )
        )
    )

    diff_model = Timings_Transformer.TimingsTransformer(
        sequence_length,
        diff_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(getDevice())
    diff_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), diff_path
            )
        )
    )

    length_model = Timings_Transformer.TimingsTransformer(
        sequence_length,
        len_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(getDevice())
    length_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), len_path
            )
        )
    )

    pitch, diff, length = midiToNotesTransformer(file_name)
    notes_to_append = Run_RNN_Model.midiToNotes(file_name)

    # Set the loaded transformer to evaluation mode
    pitch_model.eval()
    diff_model.eval()
    length_model.eval()

    # Collect the note diffs values as an array
    note_diffs = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(file_name)
    )
    note_diffs.insert(0, 0)
    note_diffs = np.asarray(note_diffs)
    print(Transformer_Timings.getMidiTempo(file_name))
    print(note_diffs)
    print(diff)

    # Collect note length values as an array
    note_lengths = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(file_name)
    )
    note_lengths = np.asarray(note_lengths)
    print(Transformer_Timings.getMidiTempo(file_name))
    print(note_lengths)

    # Create the note vocab which is used
    note_vocab = Transformer_Pitch.createVocab()

    if type_to_generate == "Notes":
        generated_notes = generateNotesGroupNumNotes(
            pitch_model,
            diff_model,
            length_model,
            pitch,
            diff,
            length,
            sequence_length,
            batch_size,
            note_diffs,
            note_lengths,
            note_vocab,
            prev_notes,
            temp=temp,
            notes_to_generate=num_to_generate,
        )
    elif type_to_generate == "Seconds":
        generated_notes = generateNotesGroupNumSeconds(
            pitch_model,
            diff_model,
            length_model,
            pitch,
            diff,
            length,
            sequence_length,
            batch_size,
            note_diffs,
            note_lengths,
            note_vocab,
            prev_notes,
            temp=temp,
            num_seconds=num_to_generate,
        )
    elif type_to_generate == "Bars":
        generated_notes = generateNotesGroupNumBars(
            pitch_model,
            diff_model,
            length_model,
            pitch,
            diff,
            length,
            sequence_length,
            batch_size,
            note_diffs,
            note_lengths,
            note_vocab,
            prev_notes,
            temp=temp,
            num_bars=num_to_generate,
            file_name=file_name,
        )

    generated_notes = pd.DataFrame(
        generated_notes, columns=("pitch", "diff", "length", "start", "end")
    )

    # Optional function, appends the generated notes to the original ones
    appended_generated_notes = Run_RNN_Model.appendNotes(
        notes_to_append, generated_notes
    )

    example_file = "example.midi"
    example_pm = Run_RNN_Model.notesToMidi(
        generated_notes, out_file=example_file, instrument_name="Acoustic Grand Piano"
    )

    example_full_file = "example_full.midi"
    example_full_pm = Run_RNN_Model.notesToMidi(
        appended_generated_notes,
        out_file=example_full_file,
        instrument_name="Acoustic Grand Piano",
    )

    # create file to refer to
    f = open("output.mid", "w")
    example_pm.write("output.mid")
    f = open("output_full.mid", "w")
    example_full_pm.write("output_full.mid")

    return generated_notes, appended_generated_notes


if __name__ == "__main__":
    # Random seed set, can be removed if needed but then will be different every time
    seed = 55
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set variables
    batch_size = 32
    embedded_heads = 30
    num_heads = 8
    num_layers = 3
    diff_values = 17
    len_values = 16
    sequence_length = 25
    vocab = 128

    # Set variables to be used in generating notes
    temp = 2
    notes_to_generate = 100
    repetition = 6
    prev_notes = []

    # loads the models
    pitch_model = Pitches_Transformer.PitchesTransformer(
        sequence_length,
        vocab,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(getDevice())
    pitch_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0], "Transformer_Model_Pitch/Transformer_Model_Pitch.pth"
            )
        )
    )

    diff_model = Timings_Transformer.TimingsTransformer(
        sequence_length,
        diff_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(getDevice())
    diff_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0], "Transformer_Model_Diff/Transformer_Model_Diff.pth"
            )
        )
    )

    length_model = Timings_Transformer.TimingsTransformer(
        sequence_length,
        len_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(getDevice())
    length_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0], "Transformer_Model_Length/Transformer_Model_Length.pth"
            )
        )
    )

    # Filename (temp for testing)
    file_name = os.path.join(os.sys.path[0], "MIDI-Examples/Apashe - Distance.mid")
    pitch, diff, length = midiToNotesTransformer(file_name)
    notes_to_append = Run_RNN_Model.midiToNotes(file_name)

    # Set the loaded transformer to evaluation mode
    pitch_model.eval()
    diff_model.eval()
    length_model.eval()

    # Collect the note diffs values as an array
    note_diffs = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(file_name)
    )
    note_diffs.insert(0, 0)
    note_diffs = np.asarray(note_diffs)
    print(Transformer_Timings.getMidiTempo(file_name))
    print(note_diffs)
    print(diff)

    # Collect note length values as an array
    note_lengths = Transformer_Timings.getNoteLengths(
        Transformer_Timings.getMidiTempo(file_name)
    )
    note_lengths = np.asarray(note_lengths)
    print(Transformer_Timings.getMidiTempo(file_name))
    print(note_lengths)

    # Create the note vocab which is used
    note_vocab = Transformer_Pitch.createVocab()

    # Generate notes based on number of notes to generate
    # generated_notes = generateNotesGroupNumNotes(pitch_model, diff_model, length_model, pitch, diff, length, sequence_length, batch_size, note_diffs, note_lengths, note_vocab,  prev_notes,temp=temp, notes_to_generate=10)
    # Generate notes based on number of seconds
    # generated_notes = generateNotesGroupNumSeconds(pitch_model, diff_model, length_model, pitch, diff, length, sequence_length, batch_size, note_diffs, note_lengths, note_vocab, prev_notes, temp=temp, num_seconds=10)
    # Generate notes based on number of bars
    generated_notes = generateNotesGroupNumBars(
        pitch_model,
        diff_model,
        length_model,
        pitch,
        diff,
        length,
        sequence_length,
        batch_size,
        note_diffs,
        note_lengths,
        note_vocab,
        prev_notes,
        temp=temp,
        num_bars=10,
        file_name=file_name,
    )

    # Format generated notes correctly
    generated_notes = pd.DataFrame(
        generated_notes, columns=("pitch", "diff", "length", "start", "end")
    )

    Run_RNN_Model.plotPianoRoll(generated_notes)

    # Optional function, appends the generated notes to the original ones
    generated_notes = Run_RNN_Model.appendNotes(notes_to_append, generated_notes)

    # Create file to write output to
    example_file = "example.midi"
    example_pm = Run_RNN_Model.notesToMidi(
        generated_notes, out_file=example_file, instrument_name="Acoustic Grand Piano"
    )

    # create file to refer to
    f = open(os.path.join(os.sys.path[0], "outputTransformer.mid"), "w")
    example_pm.write(os.path.join(os.sys.path[0], "outputTransformer.mid"))

    # Display information about the generated files
    Run_RNN_Model.plotPianoRoll(generated_notes)
    Run_RNN_Model.plotDistribution(generated_notes)
    plt.show()
