import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
from utils import *
from scipy.special import softmax
import Transformer_Timings
import Transformer_Pitch
import random
import warnings
import Run_RNN_Model
import GAN_Transformer_Pitch
import GAN_Transformer_Timing
import CNN_GAN_Pitch
import CNN_GAN_Timings
import Run_Transformer_Model

warnings.simplefilter(action="ignore", category=FutureWarning)

repetition = 6

# Run the model
def runModel(
    model: GAN_Transformer_Timing, input: np.array, batch_size: int, temp: float
):
    # Format input for model
    input = input.to_numpy()
    input = np.concatenate(input, axis=0)
    input = (
        torch.tensor(input).long().to(Run_Transformer_Model.getDevice()).unsqueeze(0)
    )
    input = input.repeat(batch_size, 1)
    input = input.swapaxes(0, 1)

    # Load input into model
    out = model.getProbabilities(input)

    # Create visualisation of model
    # dot = torchviz.make_dot(out, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    # dot.render(directory="Transformer_Model_Diff", view=True)

    # Get the prediction values from the output
    prediction = out[-1, 0, :]
    prediction = prediction.cpu().detach().numpy()
    prediction = prediction * temp
    prediction = softmax(prediction)

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

            pitch_prediction = runModel(pitch_model, pitch_input, batch_size, temp)
            diff_prediction = runModel(diff_model, diff_input, batch_size, temp)
            length_prediction = runModel(length_model, length_input, batch_size, temp)

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

            pitch_prediction = runModel(pitch_model, pitch_input, batch_size, temp)
            diff_prediction = runModel(diff_model, diff_input, batch_size, temp)
            length_prediction = runModel(length_model, length_input, batch_size, temp)

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


def initialiseCNNModels(
    sequence_length: int,
    vocab: int,
    num_heads: int,
    embedded_heads: int,
    num_layers: int,
    batch_size: int,
    diff_values: int,
    len_values: int,
    genre : str
):
    pitch_model = CNN_GAN_Pitch.Generator(
        sequence_length,
        vocab,
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    pitch_model.load_state_dict(
        torch.load(
            os.path.join(os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), "GAN_CNN/GAN_CNN_Generator_Model_Pitch" + genre + ".pth")
        )
    )

    diff_model = CNN_GAN_Timings.Generator(
        sequence_length,
        diff_values,
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    diff_model.load_state_dict(
        torch.load(os.path.join(os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), "GAN_CNN/GAN_CNN_Model_Diff" + genre + ".pth"))
    )

    length_model = CNN_GAN_Timings.Generator(
        sequence_length,
        len_values,
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    length_model.load_state_dict(
        torch.load(os.path.join(os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), "GAN_CNN/GAN_CNN_Model_Len" + genre + ".pth"))
    )

    return pitch_model, diff_model, length_model


def initialiseTransformerModels(
    sequence_length: int,
    vocab: int,
    num_heads: int,
    embedded_heads: int,
    num_layers: int,
    batch_size: int,
    diff_values: int,
    len_values: int,
    genre : str
):
    pitch_model = GAN_Transformer_Pitch.Generator(
        sequence_length,
        vocab,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    pitch_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), "GAN_Transformer/GAN_Transformer_Model_Pitch" + genre + ".pth"
            )
        )
    )

    diff_model = GAN_Transformer_Timing.Generator(
        sequence_length,
        diff_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    diff_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), "GAN_Transformer/GAN_Transformer_Model_Diff" + genre + ".pth"
            )
        )
    )

    length_model = GAN_Transformer_Timing.Generator(
        sequence_length,
        len_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    length_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), "GAN_Transformer/GAN_Transformer_Model_Len" + genre + ".pth"
            )
        )
    )

    return pitch_model, diff_model, length_model


def generateFromUI(
    file_name: str,
    num_to_generate: int,
    type_to_generate: str,
    temp: float,
    model_name: str,
):
    # Set variables
    batch_size = 32
    embedded_heads = 30
    num_heads = 8
    num_layers = 3
    diff_values = 17
    len_values = 16
    sequence_length = 32
    vocab = 128

    if model_name.__contains__("Classical"):
        genre = ""
    elif model_name.__contains__("Dance/Electronic"):
        genre = "_Electronic"
    elif model_name.__contains__("Jazz"):
        genre = "_Jazz"

    if model_name.__contains__("CNN"):
        pitch_model, diff_model, length_model = initialiseCNNModels(
            sequence_length,
            vocab,
            num_heads,
            embedded_heads,
            num_layers,
            batch_size,
            diff_values,
            len_values,
            genre
        )
    elif model_name.__contains__("Transformer"):
        pitch_model, diff_model, length_model = initialiseTransformerModels(
            sequence_length,
            vocab,
            num_heads,
            embedded_heads,
            num_layers,
            batch_size,
            diff_values,
            len_values,
            genre
        )

    # Set variables to be used in generating notes
    prev_notes = []

    pitch, diff, length = Run_Transformer_Model.midiToNotesTransformer(file_name)
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
    sequence_length = 32
    vocab = 128

    # Set variables to be used in generating notes
    temp = 2
    notes_to_generate = 100
    repetition = 6
    prev_notes = []

    # loads the models
    pitch_model = GAN_Transformer_Pitch.Generator(
        sequence_length,
        vocab,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    pitch_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Pitch.pth"
            )
        )
    )

    # Loads CNN GAN Pitch Model
    pitch_model = CNN_GAN_Pitch.Generator(
        sequence_length,
        vocab,
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    pitch_model.load_state_dict(
        torch.load(
            os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Generator_Model_Pitch.pth")
        )
    )

    diff_model = GAN_Transformer_Timing.Generator(
        sequence_length,
        diff_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    diff_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Diff.pth"
            )
        )
    )

    # Diff model CNN
    diff_model = CNN_GAN_Timings.Generator(
        sequence_length,
        diff_values,
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    diff_model.load_state_dict(
        torch.load(os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Diff.pth"))
    )

    length_model = GAN_Transformer_Timing.Generator(
        sequence_length,
        len_values,
        num_heads * embedded_heads,
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    length_model.load_state_dict(
        torch.load(
            os.path.join(
                os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Len.pth"
            )
        )
    )

    # Diff model CNN
    length_model = CNN_GAN_Timings.Generator(
        sequence_length,
        len_values,
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    length_model.load_state_dict(
        torch.load(os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Len.pth"))
    )

    # Filename (temp for testing)
    file_name = os.path.join(os.sys.path[0], "MIDI-Examples/Apashe - Distance.mid")
    pitch, diff, length = Run_Transformer_Model.midiToNotesTransformer(file_name)
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
    # generated_notes = generateNotesGroupNumNotes(pitch_model, diff_model, length_model, pitch, diff, length, sequence_length, batch_size, note_diffs, note_lengths, note_vocab,  prev_notes,temp=temp, notes_to_generate=32)
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
    f = open(os.path.join(os.sys.path[0], "outputTransformerGAN.mid"), "w")
    example_pm.write(os.path.join(os.sys.path[0], "outputTransformerGAN.mid"))

    # Display information about the generated files
    Run_RNN_Model.plotPianoRoll(generated_notes)
    Run_RNN_Model.plotDistribution(generated_notes)
    plt.show()
