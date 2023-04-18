from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.figure import *
from matplotlib.backends.backend_qt5agg import *
import threading
import Run_Transformer_GAN_Model
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import Run_RNN_Model
import Run_Transformer_Model
import pandas as pd
from typing import Optional

# Canvas to display distribution graphs
class DistributionCanvas(FigureCanvas):
    def __init__(self, parent):
        # Create figure and subplots
        fig, self.ax = plt.subplots(nrows=1, ncols=3, figsize=(5, 4), dpi=80)
        super().__init__(fig)
        self.setParent(parent)
        self.ax[0].set_title("Pitch Distribution")
        self.ax[1].set_title("Diff Distribution")
        self.ax[2].set_title("Length Distribution")

    # Add in data
    def updateData(self, notes: pd.DataFrame, drop_percentile: int = 2.5):
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[2].cla()

        # Create histogram of pitch
        sns.histplot(notes, ax=self.ax[0], x="pitch", bins=18)

        # Create histogram of diff
        max_diff = np.percentile(notes["diff"], 100 - drop_percentile)
        sns.histplot(notes, ax=self.ax[1], x="diff", bins=np.linspace(0, max_diff, 18))

        # Create histogram of length
        max_length = np.percentile(notes["length"], 100 - drop_percentile)
        sns.histplot(
            notes, ax=self.ax[2], x="length", bins=np.linspace(0, max_length, 18)
        )


# Canvas to display piano rolls
class PianoRollCanvas(FigureCanvas):
    def __init__(self, parent, title):
        # Create figure and subplots
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), dpi=80)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Pitch")

    # Add in data
    def updateData(self, notes: pd.DataFrame, count: Optional[int] = None):
        self.ax.cla()

        # Plot the pitch and start/stop of each note
        plot_pitch = np.stack([notes["pitch"], notes["pitch"]], axis=0)
        plot_start_stop = np.stack([notes["start"], notes["end"]], axis=0)
        self.ax.plot(
            plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker="."
        )
        plt.plot(
            plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker="."
        )


# Holder for Distribution Canvas
class DistributionCanvasHolder(QWidget):
    def __init__(self):
        super().__init__()

        # Create holder
        self.main = QWidget()
        layout = QVBoxLayout(self.main)

        self.chart = DistributionCanvas(self)
        layout.addWidget(NavigationToolbar(self.chart, self))
        layout.addWidget(self.chart)

        self.setLayout(layout)

    # Update data in Canvas
    def updateData(self, notes: pd.DataFrame, drop_percentile: int = 2.5):
        self.chart.updateData(notes, drop_percentile)


# Holder for Piano Roll Canvas
class PianoRollCanvasHolder(QWidget):
    def __init__(self, title):
        super().__init__()

        # Create holder
        self.main = QWidget()
        layout = QVBoxLayout(self.main)

        self.chart = PianoRollCanvas(self, title)
        layout.addWidget(NavigationToolbar(self.chart, self))
        layout.addWidget(self.chart)

        self.setLayout(layout)

    # Update data in Canvas
    def updateData(self, notes: pd.DataFrame, count: Optional[int] = None):
        self.chart.updateData(notes, count)


# MainWindow class for the GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Define class variables
        self.file_name = ""
        self.model = "RNN - Classical"
        self.num_to_generate = 10
        self.temperature = 1.0
        self.type_to_generate = "Notes"
        self.generated_notes = []
        self.appended_generated_notes = []

        # Set title
        self.setWindowTitle("MIDI Neural Network Music Generator")

        # Define layouts
        main_layout = QVBoxLayout()
        title_layout = QVBoxLayout()
        sub_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        left_selection_layout = QGridLayout()
        left_button_layout = QGridLayout()

        # Add title to title layout
        title_layout.addWidget(
            self.createLabel(
                "MIDI Neural Network Music Generator",
                font_size=36,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
        )

        # Define Left layout widgets
        self.choose_file_button = self.createButton("Choose File", font_size=14)
        self.choose_file_button.clicked.connect(self.chooseFileClicked)

        self.choose_file_label = self.createLabel(
            "No File Selected", font_size=14, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.model_combo_box = self.createComboBox(
            [
                "RNN - Classical",
                "Transformer - Classical",
                "Transformer GAN - Classical",
                "CNN GAN - Classical",
                "RNN - Dance/Electronic",
                "Transformer - Dance/Electronic",
                "Transformer GAN - Dance/Electronic",
                "CNN GAN - Dance/Electronic",
                "RNN - Jazz",
                "Transformer - Jazz",
                "Transformer GAN - Jazz",
                "CNN GAN - Jazz",
            ],
            font_size=14,
        )
        self.model_combo_box.currentTextChanged.connect(self.modelComboSelection)

        self.int_input = self.createIntInput("10", font_size=14)
        self.int_input.textChanged.connect(self.intTextChanged)

        self.temp_input = self.createDoubleInput("1.0", font_size=14)
        self.temp_input.textChanged.connect(self.tempTextChanged)

        self.generate_combo_box = self.createComboBox(
            ["Notes", "Bars", "Seconds"], font_size=14
        )
        self.generate_combo_box.currentTextChanged.connect(self.generateComboSelection)

        left_selection_layout.addWidget(self.choose_file_button, 0, 0)
        left_selection_layout.addWidget(self.choose_file_label, 0, 1)
        left_selection_layout.addWidget(
            self.createLabel(
                "Select Model/Genre",
                font_size=14,
                alignment=Qt.AlignmentFlag.AlignCenter,
            ),
            1,
            0,
        )
        left_selection_layout.addWidget(self.model_combo_box, 1, 1)
        left_selection_layout.addWidget(
            self.createLabel(
                "Temperature", font_size=14, alignment=Qt.AlignmentFlag.AlignCenter
            ),
            2,
            0,
        )
        left_selection_layout.addWidget(self.temp_input, 2, 1)
        left_selection_layout.addWidget(
            self.createLabel(
                "Model to Generate",
                font_size=14,
                alignment=Qt.AlignmentFlag.AlignCenter,
            ),
            3,
            0,
        )
        left_selection_layout.addWidget(self.int_input, 3, 1)
        left_selection_layout.addWidget(self.generate_combo_box, 3, 2)

        left_selection_widget = QWidget()
        left_selection_widget.setLayout(left_selection_layout)

        self.generate_midi_button = self.createButton("Generate MIDI", font_size=18)
        self.generate_midi_button.clicked.connect(self.generateMidiClicked)
        self.generate_midi_button.setEnabled(False)

        self.generate_midi_label = self.createLabel(
            "", font_size=14, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.save_midi_button = self.createButton("Save Generated MIDI", font_size=18)
        self.save_midi_button.clicked.connect(self.saveMidiClicked)
        self.save_midi_button.setEnabled(False)

        self.save_full_midi_button = self.createButton("Save Full MIDI", font_size=18)
        self.save_full_midi_button.clicked.connect(self.saveFullMidiClicked)
        self.save_full_midi_button.setEnabled(False)

        self.play_midi_button = self.createButton("Play Generated MIDI", font_size=18)
        self.play_midi_button.clicked.connect(self.playMidiClicked)
        self.play_midi_button.setEnabled(False)

        self.play_full_midi_button = self.createButton("Play Full MIDI", font_size=18)
        self.play_full_midi_button.clicked.connect(self.playFullMidiClicked)
        self.play_full_midi_button.setEnabled(False)

        left_button_layout.addWidget(self.save_midi_button, 0, 0)
        left_button_layout.addWidget(self.save_full_midi_button, 0, 1)
        left_button_layout.addWidget(self.play_midi_button, 1, 0)
        left_button_layout.addWidget(self.play_full_midi_button, 1, 1)

        left_button_widget = QWidget()
        left_button_widget.setLayout(left_button_layout)

        self.distribution_graph = DistributionCanvasHolder()

        left_layout.addWidget(
            self.createLabel(
                "Select MIDI File:",
                font_size=16,
                alignment=Qt.AlignmentFlag.AlignCenter,
            ),
            1,
        )
        left_layout.addWidget(left_selection_widget, 3)
        left_layout.addWidget(self.generate_midi_button, 1)
        left_layout.addWidget(self.generate_midi_label, 1)
        left_layout.addWidget(left_button_widget, 2)
        left_layout.addWidget(
            self.createLabel(
                "Note Distribution:",
                font_size=20,
                alignment=Qt.AlignmentFlag.AlignCenter,
            ),
            1,
        )
        left_layout.addWidget(self.distribution_graph, 10)

        self.generated_midi_graph = PianoRollCanvasHolder("Generated MIDI")
        self.full_midi_graph = PianoRollCanvasHolder("Full MIDI")

        # Define right layout widgets
        right_layout.addWidget(
            self.createLabel(
                "Piano Roll of Generated MIDI:",
                font_size=20,
                alignment=Qt.AlignmentFlag.AlignCenter,
            ),
            1,
        )
        right_layout.addWidget(self.generated_midi_graph, 10)
        right_layout.addWidget(
            self.createLabel(
                "Piano Roll of Full MIDI:",
                font_size=20,
                alignment=Qt.AlignmentFlag.AlignCenter,
            ),
            1,
        )
        right_layout.addWidget(self.full_midi_graph, 10)

        # Format layouts
        main_layout.addLayout(title_layout, 1)
        main_layout.addLayout(sub_layout, 10)
        sub_layout.addLayout(left_layout, 1)
        sub_layout.addLayout(right_layout, 1)

        # Add widget to MainWindow
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    # Create a label with passed values
    def createLabel(self, label_text: str, font_size: int, alignment: Qt.AlignmentFlag):
        label = QLabel(label_text)

        font = label.font()
        font.setPointSize(font_size)
        label.setFont(font)

        label.setAlignment(alignment)

        return label

    # Create a button with passed values
    def createButton(self, button_text: str, font_size: int):
        button = QPushButton(button_text)

        font = button.font()
        font.setPointSize(font_size)
        button.setFont(font)

        return button

    # Create a combo box
    def createComboBox(self, values: list, font_size: int):
        combo_box = QComboBox()
        combo_box.addItems(values)

        font = combo_box.font()
        font.setPointSize(font_size)
        combo_box.setFont(font)

        return combo_box

    # Create int input
    def createIntInput(self, default_value: str, font_size: int):
        line_edit = QLineEdit()
        line_edit.setMaxLength(3)
        line_edit.setPlaceholderText(default_value)

        # Create a validator to allow for only int input
        int_validator = QIntValidator()
        int_validator.setRange(1, 999)
        line_edit.setValidator(int_validator)

        font = line_edit.font()
        font.setPointSize(font_size)
        line_edit.setFont(font)

        return line_edit

    # Create double input
    def createDoubleInput(self, default_value: str, font_size: int):
        line_edit = QLineEdit()
        line_edit.setMaxLength(4)
        line_edit.setPlaceholderText(default_value)

        # Create a validator to allow for only int input
        double_validator = QDoubleValidator()
        double_validator.setRange(1.0, 999.0, 3)
        line_edit.setValidator(double_validator)

        font = line_edit.font()
        font.setPointSize(font_size)
        line_edit.setFont(font)

        return line_edit

    # Choose a file when clicked
    @pyqtSlot()
    def chooseFileClicked(self):
        print("CHOOSE FILE CLICKED")

        file_name = QFileDialog.getOpenFileName(
            self,
            "Select Midi File",
            "${HOME}",
            "Midi Files (*.mid)",
        )

        self.file_name = file_name.__getitem__(0)

        if self.file_name == "":
            print("NO FILE SELECTED")
            choose_file_label_text = "No File Selected"
            self.generate_midi_button.setEnabled(False)
        else:
            print(self.file_name)
            choose_file_label_text = "File Selected:\n" + file_name.__getitem__(0)
            self.generate_midi_button.setEnabled(True)

        self.choose_file_label.setText(choose_file_label_text)

    # Update model selected when combo box option clicked
    def modelComboSelection(self, selection: str):
        self.model = selection

    # Update model variables when integer text updated
    def intTextChanged(self, text: str):
        if text == "" or int(text) == 0:
            self.num_to_generate = 10
        else:
            self.num_to_generate = int(text)

    # Update temperature variable
    def tempTextChanged(self, text: str):
        if text == "" or text == "." or float(text) == 0.0:
            self.temperature = 1.0
        else:
            self.temperature = float(text)

    # Update combo selection
    def generateComboSelection(self, selection: str):
        self.type_to_generate = selection

    # Generate notes
    def generateMidiClicked(self):
        self.generate_midi_label.setText("Generating Midi...")
        thread = threading.Thread(target=self.generateMidiThread, args=())
        thread.start()

    # Save the generated midi files
    def saveMidiClicked(self):
        print("save midi clicked")

        file_name = QFileDialog.getSaveFileName(
            self, "Save File", "${HOME}", "Midi Files (*.mid)"
        )

        file_name = file_name.__getitem__(0)

        if file_name == "":
            print("NO FILE SELECTED")
        else:
            generated_notes = self.generated_notes

            example_file = "example.midi"
            example_pm = Run_RNN_Model.notesToMidi(
                generated_notes,
                out_file=example_file,
                instrument_name="Acoustic Grand Piano",
            )

            f = open("output.mid", "w")
            example_pm.write(filename=file_name)

            self.generate_midi_label.setText("Midi File Saved")

    # Play generated MIDI file
    def playMidiClicked(self):
        thread = threading.Thread(target=self.playMusicThread, args=("output.mid",))
        thread.start()

    # Save the full midi
    def saveFullMidiClicked(self):
        print("save midi clicked")

        file_name = QFileDialog.getSaveFileName(
            self, "Save File", "${HOME}", "Midi Files (*.mid)"
        )

        file_name = file_name.__getitem__(0)

        if file_name == "":
            print("NO FILE SELECTED")
        else:
            appended_generated_notes = self.appended_generated_notes

            example_file = "example.midi"
            example_pm = Run_RNN_Model.notesToMidi(
                appended_generated_notes,
                out_file=example_file,
                instrument_name="Acoustic Grand Piano",
            )

            f = open("output.mid", "w")
            example_pm.write(filename=file_name)

            self.generate_midi_label.setText("Midi File Saved")

    # Play the full MIDI
    def playFullMidiClicked(self):
        print("Play midi clicked")
        thread = threading.Thread(
            target=self.playMusicThread, args=("output_full.mid",)
        )
        thread.start()

    # Generate MIDI in a new thread to prevent UI blocking
    def generateMidiThread(self):
        if self.model.__contains__("Classical"):
            genre = "Classical"
        elif self.model.__contains__("Dance/Electronic"):
            genre = "Dance/Electronic"
        elif self.model.__contains__("Jazz"):
            genre = "Jazz"

        if self.model.__contains__("RNN"):
            (
                self.generated_notes,
                self.appended_generated_notes,
            ) = Run_RNN_Model.generateFromUI(
                self.file_name,
                self.num_to_generate,
                self.type_to_generate,
                self.temperature,
                genre
            )
        elif (
            self.model.__contains__("Transformer GAN")
            or self.model.__contains__("CNN GAN")
        ):
            (
                self.generated_notes,
                self.appended_generated_notes,
            ) = Run_Transformer_GAN_Model.generateFromUI(
                self.file_name,
                self.num_to_generate,
                self.type_to_generate,
                self.temperature,
                self.model,
            )
        elif self.model.__contains__("Transformer"):
            (
                self.generated_notes,
                self.appended_generated_notes,
            ) = Run_Transformer_Model.generateFromUI(
                self.file_name,
                self.num_to_generate,
                self.type_to_generate,
                self.temperature,
                genre
            )

        self.generate_midi_label.setText("Midi Generated!")

        # Enable buttons when MIDI has been generated
        self.save_midi_button.setEnabled(True)
        self.play_midi_button.setEnabled(True)
        self.save_full_midi_button.setEnabled(True)
        self.play_full_midi_button.setEnabled(True)

        self.updateGraphs()

    # Update the graphs when midi generated
    def updateGraphs(self):
        self.distribution_graph.updateData(self.generated_notes, drop_percentile=2.5)
        self.generated_midi_graph.updateData(self.generated_notes)
        self.full_midi_graph.updateData(self.appended_generated_notes)

    # Thread to playback MIDI
    def playMusicThread(self, song: str):
        Run_RNN_Model.playMusic(song)


if __name__ == "__main__":
    # Define the application
    app = QApplication([])

    # Create a window of MainWindow class and show it
    window = MainWindow()
    window.show()

    # Run the app
    app.exec()
