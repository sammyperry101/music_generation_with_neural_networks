import collections
import Timing_Comparator
import mido
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import pygame
from matplotlib import pyplot as plt
from typing import Optional
import os
import Note_Matcher
import RNN_Model

# This method converts a midi file into an array of notes
def midiToNotes(midi_file: str):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes["pitch"].append(note.pitch)
        notes["diff"].append(start - prev_start)
        notes["length"].append(end - start)
        notes["start"].append(start)
        notes["end"].append(end)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


# This method is used to predict the next note in the sequence
# First need to provide a starting sequence
# This function generates one note from a sequence
# Temperature controls randomness of geneerated notes
def predict_next_note(
    notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0
):

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    # Get the predicted pitch, diff and len values
    predictions = model.predict(inputs)
    pitch_logits = predictions["pitch"]
    diff = predictions["diff"]
    length = predictions["length"]

    # Apply temp and select pitch, diff and len based on predicted values
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    print(pitch)
    pitch = tf.squeeze(pitch, axis=-1)
    print(pitch)
    length = tf.squeeze(length, axis=-1)
    diff = tf.squeeze(diff, axis=-1)

    # Make diff and len positive values
    diff = tf.maximum(-diff, diff)
    length = tf.maximum(-length, length)

    return int(pitch), float(diff), float(length)


# This allows you to create midi based on notes generated
def notesToMidi(
    notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100
):
    # Create PrettyMIDI object and instrument
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    # Iterate through notes and create midi Notes to append to the PrettyMIDI object
    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note["diff"])
        end = float(start + note["length"])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note["pitch"]),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    # Write result to a file
    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


# This code allows you to play any midi files at any point
def playMusic(music_file: str):
    # Set variables
    frequency = 44100
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(frequency, bitsize, channels, buffer)

    # Create clock and use pygame to playback file
    clock = pygame.time.Clock()
    print("Music file", music_file)
    pygame.mixer.music.load(music_file)

    # Play file
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(25)


# Method plots the piano roll and number of notes it wants to use
def plotPianoRoll(notes: pd.DataFrame, count: Optional[int] = None):
    # Create title
    title = "Piano Roll"
    # Define figure
    plt.figure(figsize=(20, 4))

    # Plot the pitch and start/stop of each note
    plot_pitch = np.stack([notes["pitch"], notes["pitch"]], axis=0)
    plot_start_stop = np.stack([notes["start"], notes["end"]], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")

    # Add labels
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch")

    # Apply title and show graph
    _ = plt.title(title)
    plt.show()


# This shows distribution of notes
def plotDistribution(notes: pd.DataFrame, drop_percentile: int = 2.5):
    # Create figure and subplots
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    # Create histogram of pitch
    sns.histplot(notes, x="pitch", bins=18)

    plt.subplot(1, 3, 2)
    # Create histogram of diff
    max_diff = np.percentile(notes["diff"], 100 - drop_percentile)
    sns.histplot(notes, x="diff", bins=np.linspace(0, max_diff, 18))

    plt.subplot(1, 3, 3)
    # Create histogram of length
    max_length = np.percentile(notes["length"], 100 - drop_percentile)
    sns.histplot(notes, x="length", bins=np.linspace(0, max_length, 18))
    # Show graphs
    plt.show()


# Method to generate the set of notes desired using the model
# Generates based on the passed number of notes desired, default is 100
def generateNoteGroupNumNotes(
    model: tf.keras.Model,
    notes: np.array,
    file_name: str,
    temperature: float = 1.0,
    num_predictions: int = 100,
):
    # encode notes using same method as before
    key_order = ["pitch", "diff", "length"]
    sample_notes = np.stack([notes[key] for key in key_order], axis=1)

    # The initial sequence of notes
    # Pitch is normalized similar to training sequences
    # Only set of notes equal to model input shape is selected for song
    input_notes = sample_notes[: model.input_shape[1]] / np.array([128, 1, 1])

    # Iterate through until correct amount of notes generated
    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        # Predict note and get returned values
        pitch, diff, length = predict_next_note(input_notes, model, temperature)

        # Round diff and length timings based on tempo of selected piece
        diff = roundDiffTimings(file_name, diff)
        length = roundLengthTimings(file_name, length)

        # Update timings of note based on previous note timings
        start = prev_start + diff
        end = start + length
        input_note = (pitch, diff, length)
        generated_notes.append((*input_note, start, end))

        # Remove the first note of sequence and apped new note to ensure notes generated are based on new sequence
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)

        prev_start = start

    return generated_notes, key_order


# Method to generate the set of notes desired using the model
# Generates based on the passed number of seconds, default is 10
def generateNoteGroupNumSeconds(
    model: tf.keras.Model,
    notes: np.array,
    file_name: str,
    temperature: float = 1.0,
    num_seconds: int = 10,
):
    # encode notes using same method as before
    key_order = ["pitch", "diff", "length"]
    sample_notes = np.stack([notes[key] for key in key_order], axis=1)

    # The initial sequence of notes
    # pitch is normalized similar to training sequences
    # only set of notes equal to model input shape is selected for song
    input_notes = sample_notes[: model.input_shape[1]] / np.array([128, 1, 1])

    # Iterate through until correct amount of notes generated
    generated_notes = []
    prev_start = 0
    end = 0
    while end < num_seconds:
        # Predict note and get returned values
        pitch, diff, length = predict_next_note(input_notes, model, temperature)

        # Round diff and length timings based on tempo of selected piece
        diff = roundDiffTimings(file_name, diff)
        length = roundLengthTimings(file_name, length)

        # Update timings of note based on previous note timings
        start = prev_start + diff
        end = start + length
        input_note = (pitch, diff, length)
        generated_notes.append((*input_note, start, end))

        # Remove the first note of sequence and apped new note to ensure notes generated are based on new sequence
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)

        prev_start = start

    return generated_notes, key_order


# Method to generate the set of notes desired using the model
# Generates based on the passed number of bars, default is 8
def generateNoteGroupNumBars(
    model: tf.keras.Model,
    notes: np.array,
    file_name: str,
    temperature: float = 1.0,
    num_bars: int = 8,
):
    return generateNoteGroupNumSeconds(
        model,
        notes,
        file_name,
        temperature,
        (num_bars / (getMidiTempo(file_name) / 4)) * 60,
    )


# Checks for tempo information in midi file
# If none, then returns default tempo (120 bpm)
def getMidiTempo(file_name: str):
    midi = mido.MidiFile(file_name)
    for msg in midi:
        if msg.type == "set_tempo":
            return int(round(mido.tempo2bpm(msg.tempo), 0))

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
    # Calculate value of crotchet based on tempo
    crotchet = round(60 / tempo, 2)
    note_lengths = []

    # Round update the timings values based on crotchet value (in seconds)
    for i in timings:
        note_lengths.append(round(i * crotchet, 2))

    return note_lengths


# Given a time value, round it to nearest valid length
# given the list of valid times
def roundLengthTimings(file_name: str, time: float):
    note_lengths = getNoteLengths(getMidiTempo(file_name))
    note_lengths = np.asarray(note_lengths)
    i = (np.abs(note_lengths - time)).argmin()
    return note_lengths[i]


# Given a time value, round it to nearest valid diff
# given the list of valid times
# Difference is that 0 is a valid time for diff, NOT for
# length as cannot have notes of 0 length
def roundDiffTimings(file_name: str, time: float):
    note_lengths = getNoteLengths(getMidiTempo(file_name))
    note_lengths.insert(0, 0)
    note_lengths = np.asarray(note_lengths)
    i = (np.abs(note_lengths - time)).argmin()
    return note_lengths[i]


# Append the new generated notes to the end of the set of original input notes
def appendNotes(start_notes: pd.DataFrame, end_notes: pd.DataFrame):
    new_notes = start_notes
    # Get the end value for the final note of the piece as starting value
    prev_start = new_notes["end"].iloc[-1]

    # Iterate through all new notes
    for note in end_notes.iloc:
        # Get the pitch, diff and length value of notes to be appended
        pitch = note["pitch"]
        diff = note["diff"]
        length = note["length"]

        # Calculate start and end of the notes
        start = prev_start + diff
        end = start + length

        prev_start = start

        # Create new note object
        new_note = {
            "pitch": pitch,
            "diff": diff,
            "length": length,
            "start": start,
            "end": end,
        }

        # Append note to new_notes
        new_notes = new_notes.append(new_note, ignore_index=True)

    return new_notes


# Generate notes with chosen model from UI
def generateFromUI(
    file_name: str, num_to_generate: int, type_to_generate: str, temperature: float, genre : str
):
    print("GENERATE RNN")

    if genre == "Classical":
        model_path = "RNN-Model"
    elif genre == "Dance/Electronic":
        model_path = "RNN-Electronic"
    elif genre == "Jazz":
        model_path = "RNN-Jazz"

    # Load model
    model = tf.keras.models.load_model(
        os.path.join(os.sys.path[0].replace('\\MIDIGenerator\\base_library.zip',''), model_path),
        custom_objects={"meanSquareErr": RNN_Model.meanSquareErr},
    )

    # Load midi file to create from
    notes = midiToNotes(file_name)

    # Generate notes based on num notes, seconds or bars
    if type_to_generate == "Notes":
        generated_notes, key_order = generateNoteGroupNumNotes(
            model,
            notes,
            file_name,
            temperature=temperature,
            num_predictions=num_to_generate,
        )
    elif type_to_generate == "Seconds":
        generated_notes, key_order = generateNoteGroupNumSeconds(
            model,
            notes,
            file_name,
            temperature=temperature,
            num_seconds=num_to_generate,
        )
    elif type_to_generate == "Bars":
        generated_notes, key_order = generateNoteGroupNumBars(
            model, notes, file_name, temperature=temperature, num_bars=num_to_generate
        )

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, "start", "end")
    )

    appended_generated_notes = appendNotes(notes, generated_notes)

    example_file = "example.midi"
    example_pm = notesToMidi(
        generated_notes, out_file=example_file, instrument_name="Acoustic Grand Piano"
    )

    example_full_file = "example_full.midi"
    example_full_pm = notesToMidi(
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
    # file_name_start = os.path.join(os.sys.path[0], "Evaluation Data/Jazz - I Can't Get Started First 8 Bars.mid")
    # file_name_continuation = os.path.join(os.sys.path[0], "Evaluation Data/Jazz - I Can't Get Started Continuation 4 Bars.mid")
    # file_name_generated = os.path.join(os.sys.path[0], "Evaluation Data/Generated Files/Jazz - Transformer.mid")

    # notes_start = midiToNotes(file_name_start)
    # notes_continuation = midiToNotes(file_name_continuation)
    # note_generated = midiToNotes(file_name_generated)

    # note_matcher = Note_Matcher.NoteMatcher()
    # timing_comparator = Timing_Comparator.TimingComparator()

    # print("In Key:", note_matcher.matchNotes("B♭ Major", note_generated["pitch"]))
    # print("Timing Mean Length", note_generated["length"].mean())
    # print("Timing Diff Mean:", note_generated["diff"].mean())

    # note_generated = appendNotes(notes_continuation, note_generated)

    # plotPianoRoll(note_generated)

    # Filename (temp for testing)
    file_name = os.path.join(os.sys.path[0], "MIDI-Examples/Apashe - Distance.mid")

    # Load model
    new_model = tf.keras.models.load_model(
        os.path.join(os.sys.path[0], "RNN-Model"),
        custom_objects={"meanSquareErr": RNN_Model.meanSquareErr},
    )

    # Load midi file to create from
    notes = midiToNotes(file_name)

    # Generate notes based on num notes, seconds or bars
    generated_notes, key_order = generateNoteGroupNumNotes(
        new_model, notes, file_name, temperature=1, num_predictions=10
    )
    # generated_notes, key_order = generateNoteGroupNumSeconds(new_model, notes, file_name, temperature=1, num_seconds=10)
    # generated_notes, key_order = generateNoteGroupNumBars(new_model, notes, file_name, temperature=1, num_bars=10)

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, "start", "end")
    )

    plotPianoRoll(generated_notes)

    # Test matching of note pitches on NoteMatcher
    noteMatcher = Note_Matcher.NoteMatcher()
    pitch_to_match = generated_notes["pitch"]
    match_rate = noteMatcher.matchNotes("G Minor", pitch_to_match)
    print(match_rate)

    # Test comparing timings using TimingComparator class
    timingComparator = Timing_Comparator.TimingComparator()
    length_primer = notes["length"]
    length_generated = generated_notes["length"]
    mean_diff = timingComparator.getMeanDifference(length_primer, length_generated)
    print(mean_diff)
    std_diff = timingComparator.getStdDifference(length_primer, length_generated)
    print(std_diff)
    var_diff = timingComparator.getVarDifference(length_primer, length_generated)
    print(var_diff)

    # Optional function, appends the generated notes to the original ones
    generated_notes = appendNotes(notes, generated_notes)

    example_file = "example.midi"
    example_pm = notesToMidi(
        generated_notes, out_file=example_file, instrument_name="Acoustic Grand Piano"
    )

    # create file to refer to
    f = open(os.path.join(os.sys.path[0], "output.mid"), "w")
    example_pm.write(os.path.join(os.sys.path[0], "output.mid"))

    # Generate graphs
    plotPianoRoll(generated_notes)
    plotDistribution(generated_notes)
    plt.show()

    # Plot a model of the developed LSTM-RNN Model
    # tf.keras.utils.plot_model(new_model, to_file='LSTM-Model.png', show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True, expand_nested=True)

    # # Below Code plays midi output
    try:
        playMusic(os.path.join(os.sys.path[0], "output.mid"))
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
