import torch
from torch import nn
import glob
import numpy as np
import pathlib
import os
import torch
import torch.nn as nn
from utils import *
import RNN_Model
import Run_Transformer_Model
import Pitches_Dataset
import math
import Transformer_Pitch
import GAN_Transformer_Pitch

# Generator class for Pitches
class Generator(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_tokens: int,
        num_inputs: int,
        num_layers: int,
        batch_size: int = 32,
    ):
        super().__init__()

        # Define all the layers for the CNN to use
        self.conv = nn.Conv1d(
            in_channels=batch_size,
            out_channels=batch_size // 2,
            kernel_size=sequence_length // 4 + 1,
            stride=1,
        )
        self.pooling = nn.AvgPool1d(kernel_size=sequence_length // 16 + 1)
        self.conv2 = nn.Conv1d(
            in_channels=batch_size // 2,
            out_channels=batch_size // 4,
            kernel_size=1,
            stride=1,
        )
        self.pooling2 = nn.AvgPool1d(kernel_size=sequence_length // 32 + 1)
        self.conv3 = nn.Conv1d(
            in_channels=batch_size // 4,
            out_channels=batch_size,
            kernel_size=3,
            stride=1,
        )
        self.linear_act = nn.Linear(36, 128)

        # Initialise class variables
        positional_embedding = torch.randn(
            (sequence_length, 1, 3),
            device=Run_Transformer_Model.getDevice(),
            requires_grad=True,
        )
        self.pos_embeds = positional_embedding.repeat(1, batch_size, 1)
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.encoding = nn.Embedding(num_tokens, num_inputs - 3)

        # Define batch normalisation method
        self.batch_norm = torch.nn.ModuleList(
            [nn.BatchNorm1d(sequence_length) for _ in range(num_layers)]
        )

    # Method defines how to proceed for each layer
    def forward(self, input: np.array):
        input = self.encoding(input) * math.sqrt(self.num_inputs)

        # Add positional embeddings for the input
        input = torch.cat([input, self.pos_embeds], axis=2)

        # Normalise the batches
        for layer in range(self.num_layers):
            input = input.swapaxes(0, 1)
            input = self.batch_norm[layer](input)
            input = input.swapaxes(0, 1)

        # Apply all layers of the CNN structure
        out = self.conv(input)
        out = self.pooling(out)
        out = self.conv2(out)
        out = self.pooling2(out)
        out = self.conv3(out)
        out = self.linear_act(out)
 

        # Get selected note from one-hot encoding
        all_notes = []
        for x in out:
            current_notes = []
            for y in x:
                current_max = -1000
                max_index = 0
                current_index = 0
                for z in y:
                    if z.item() > current_max:
                        current_max = z.item()
                        max_index = current_index
                    current_index += 1
                current_notes.append(max_index)
            all_notes.append(current_notes)

        return torch.Tensor(all_notes)

    # Method to return probabilty values for generating notes on user input
    def getProbabilities(self, input: np.array):
        input = self.encoding(input) * math.sqrt(self.num_inputs)

        # Add positional embeddings for the input
        input = torch.cat([input, self.pos_embeds], axis=2)

        # Normalise the batches
        for layer in range(self.num_layers):
            input = input.swapaxes(0, 1)
            input = self.batch_norm[layer](input)
            input = input.swapaxes(0, 1)

        # Apply all layers of CNN
        out = self.conv(input)
        out = self.pooling(out)
        out = self.conv2(out)
        out = self.pooling2(out)
        out = self.conv3(out)
        out = self.linear_act(out)

        return out


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
    files_to_parse = 1
    all_pitches, all_diff, all_length = Transformer_Pitch.parseFiles(
        file_names, files_to_parse
    )

    # Create the vocabulary (one-hot encoding) for the pitches
    vocab = Transformer_Pitch.createVocab()

    # Define dataset and sequence length (default is 25)
    sequence_length = 32
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
    epochs = 1

    # Build generator and define loss function & optimisation method
    generator = Generator(
        sequence_length,
        len(vocab),
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    loss_function = nn.BCEWithLogitsLoss()
    optimiser_generator = torch.optim.Adam(generator.parameters(), lr=0.001)

    # Build discriminator and define optimisation method
    discriminator = GAN_Transformer_Pitch.Discriminator().to(
        Run_Transformer_Model.getDevice()
    )
    optimiser_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

    # Set both models to train
    generator.train()
    discriminator.train()

    (
        generator,
        discriminator,
        generator_loss,
        discriminator_loss,
        batch_history,
    ) = GAN_Transformer_Pitch.trainModel(
        generator,
        discriminator,
        loss_function,
        optimiser_discriminator,
        optimiser_generator,
        dataset,
        epochs,
        batch_size,
        sequence_length,
    )

    # Plot model performance
    GAN_Transformer_Pitch.plotPerformance(
        generator_loss, discriminator_loss, batch_history
    )

    # Saves the models
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Generator_Model_Pitch.pth"),
    # )

    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Pitch.pth"),
    # )

    torch.save(
        generator.state_dict(),
        os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Generator_Model_Pitch_Electronic.pth"),
    )

    torch.save(
        discriminator.state_dict(),
        os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Pitch_Electronic.pth"),
    )

    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Generator_Model_Pitch_Jazz.pth"),
    # )

    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Pitch_Jazz.pth"),
    # )

    print("Model saved")
