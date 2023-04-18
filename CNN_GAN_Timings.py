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
import math
import Timings_Dataset
import Transformer_Timings
import GAN_Transformer_Pitch
import GAN_Transformer_Timing

# Generator class for Timings
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

        # Define all layers for CNN to use
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
        self.linear_act = nn.Linear(36, num_tokens)

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

        # Define batch normalisation for the data
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

        # Apply all layers of the CNN
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
    all_pitches, all_diff, all_length = Transformer_Timings.parseFiles(
        file_names, files_to_parse
    )

    print(all_diff)
    print(all_length)

    # Set variables
    batch_size = 32
    embedded_heads = 30
    num_heads = 8
    num_layers = 3
    diff_values = 17
    len_values = 16

    # Define dataset and sequence length (default is 25)
    sequence_length = 32
    # diff dataset
    dataset = Timings_Dataset.TimingsDataset(
        all_diff, sequence_length, files_to_parse, diff_values
    )
    # Code for length_dataset
    # dataset = Timings_Dataset.TimingsDataset(all_length, sequence_length, files_to_parse, len_values)

    # Set epochs (default is 10)
    epochs = 1

    # Build generator and define loss function & optimisation method
    # diff generator
    generator = Generator(
        sequence_length,
        diff_values,
        num_heads * embedded_heads,
        num_layers,
        batch_size=batch_size,
    ).to(Run_Transformer_Model.getDevice())
    # length generator
    # generator = Generator(sequence_length, len_values, num_heads*embedded_heads, num_layers, batch_size=batch_size).to(Run_Transformer_Model.getDevice())

    loss_function = nn.BCEWithLogitsLoss()
    optimiser_generator = torch.optim.Adam(generator.parameters(), lr=0.001)

    # Build discriminator and define optimisation method
    discriminator = GAN_Transformer_Pitch.Discriminator(batch_size, sequence_length).to(
        Run_Transformer_Model.getDevice()
    )
    optimiser_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.000001)

    # Set both models to train
    generator.train()
    discriminator.train()

    (
        generator,
        discriminator,
        generator_loss,
        discriminator_loss,
        batch_history,
    ) = GAN_Transformer_Timing.trainModel(
        generator,
        discriminator,
        loss_function,
        optimiser_discriminator,
        optimiser_generator,
        dataset,
        epochs,
        batch_size,
        sequence_length,
        diff_values,
        len_values,
    )

    # Plot model performance
    GAN_Transformer_Pitch.plotPerformance(
        generator_loss, discriminator_loss, batch_history
    )

    # Saves the models
    # Save diff Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Diff.pth"),
    # )

    # Save Len Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Len.pth"
    #     ),
    # )

    # Save Diff Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Diff.pth"),
    # )

    # Save Len Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Len.pth"
    #     ),
    # )


    torch.save(
        generator.state_dict(),
        os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Diff_Electronic.pth"),
    )

    # Save Len Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Len_Electronic.pth"
    #     ),
    # )

    # Save Diff Discriminator
    torch.save(
        discriminator.state_dict(),
        os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Diff_Electronic.pth"),
    )

    # Save Len Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Len_Electronic.pth"
    #     ),
    # )


    # Save diff Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Diff_Jazz.pth"),
    # )

    # Save Len Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_CNN/GAN_CNN_Model_Len_Jazz.pth"
    #     ),
    # )

    # Save Diff Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Diff_Jazz.pth"),
    # )

    # Save Len Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_CNN/GAN_CNN_Discriminator_Model_Len_Jazz.pth"
    #     ),
    # )

    print("Model saved")
