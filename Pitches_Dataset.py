import numpy as np
import torch
import torch.nn as nn
from utils import *

# Create a Dataset class which will be loaded into the PyTorch DataLoader
class PitchesDataset(torch.utils.data.Dataset):
    # Init method defines all variables and encodes the pitches for each
    # individual song and saves these to the all_encoded_pitches variable
    def __init__(
        self,
        vocab: list,
        all_pitches: int,
        sequence_length: int = 25,
        num_pieces: int = 5,
    ):
        self.sequence_length = sequence_length
        self.num_pieces = num_pieces
        # Number of training examples per piece
        self.training_examples = {}
        # Total number of training examples
        self.total_training_examples = 0
        self.all_pitches = all_pitches
        self.vocab = vocab

        # Iterate through all song pitches and save information on each song
        i = 0
        for pitches in all_pitches:
            self.training_examples[i] = pitches.size - (self.sequence_length + 1)
            self.total_training_examples += pitches.size - (self.sequence_length + 1)
            i += 1

    # Length is defined as the total number of available training examples
    def __len__(self):
        return self.total_training_examples

    # This method returns a given training example
    def __getitem__(self, index: int):
        # Find correct piece based on index passed into function
        piece_index = 0
        for i in self.training_examples:
            current = self.training_examples[i]
            if self.training_examples[i] < index:
                piece_index += 1
                index -= current
            else:
                break

        # Get the pitches for the selected piece
        pitches = self.all_pitches[piece_index]

        # Format this into input (x) and expected output (y)
        x = pitches[index : index + self.sequence_length]
        y = pitches[index + 1 : index + self.sequence_length + 1]

        x = x.to_numpy()
        y = y.to_numpy()

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        # Convert items to tensors in PyTorch and make output into one-hot encoding
        x = torch.tensor(x).long()
        y = torch.tensor(y).long()
        y = nn.functional.one_hot(y, num_classes=len(self.vocab))

        return x, y
