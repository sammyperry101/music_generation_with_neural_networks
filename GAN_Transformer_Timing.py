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
import Pitches_Multihead_Attention
import Timings_Dataset
import Transformer_Timings
import GAN_Transformer_Pitch

# Generator class for Timings
class Generator(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_tokens: int,
        num_inputs: int,
        num_heads: int,
        num_layers: int,
        batch_size: int = 32,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Initialise class variables
        positional_embedding = torch.randn(
            (sequence_length, 1, 3),
            device=Run_Transformer_Model.getDevice(),
            requires_grad=True,
        )
        self.pos_embeds = positional_embedding.repeat(1, batch_size, 1)
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.decoder = nn.Linear(num_inputs, num_tokens)
        self.encoding = nn.Embedding(num_tokens, num_inputs - 3)

        # Define batch normalisation
        self.batch_norm = torch.nn.ModuleList(
            [nn.BatchNorm1d(sequence_length) for _ in range(num_layers)]
        )

        # Define transformer attention usig Multihead Attention classes
        self.transformer_attention = torch.nn.ModuleList(
            [
                Pitches_Multihead_Attention.PitchesMultiheadAttention(
                    num_inputs, sequence_length, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    # Method defines how to proceed for each layer
    def forward(self, input: np.array):
        input = self.encoding(input) * math.sqrt(self.num_inputs)

        # Add positional embeddings for the input
        input = torch.cat([input, self.pos_embeds], axis=2)

        # Run over the layers (heads) in the multi
        for layer in range(self.num_layers):
            input = input.swapaxes(0, 1)
            input = self.batch_norm[layer](input)
            input = input.swapaxes(0, 1)
            input = self.transformer_attention[layer](input)
            input = input.swapaxes(0, 1)

        # Decode the output
        output = self.decoder(input)

        # Get selected note from one-hot encoding
        all_notes = []
        for x in output:
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

        # Run over the layers (heads) in the multi
        for layer in range(self.num_layers):
            input = input.swapaxes(0, 1)
            input = self.batch_norm[layer](input)
            input = input.swapaxes(0, 1)
            input = self.transformer_attention[layer](input)
            input = input.swapaxes(0, 1)

        # Decode the output
        output = self.decoder(input)

        return output


# Method to train the model
def trainModel(
    generator: Generator,
    discriminator: GAN_Transformer_Pitch.Discriminator,
    loss_function: nn.BCEWithLogitsLoss(),
    optimiser_discriminator: torch.optim,
    optimiser_generator: torch.optim,
    dataset: Pitches_Dataset.PitchesDataset,
    epochs: int = 10,
    batch_size: int = 32,
    sequence_length: int = 32,
    diff_values: int = 17,
    len_values: int = 16,
):
    # Create loss variables to track loss results of generator and discriminator
    generator_loss = []
    discriminator_loss = []
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
        for _, (real_data, _) in enumerate(data_loader):
            # Data labels for real data
            real_data_labels = torch.ones((batch_size, 1))
            # Samples of latent data to train Discriminator
            # Latent data sample for diff
            latent_data_samples = torch.randint(
                0, diff_values, (sequence_length, batch_size)
            )
            # Latent data sample for length
            # latent_data_samples = torch.randint(0, len_values, (sequence_length, batch_size))

            # Generate data using the generator
            generated_data = generator(latent_data_samples)
            generated_data_labels = torch.zeros((batch_size, 1))

            # Concatenace data from real and generated samples and data labels
            all_data = torch.cat((real_data, generated_data))
            all_labels = torch.cat((real_data_labels, generated_data_labels))

            # Train discriminator
            discriminator.zero_grad()
            output_disc = discriminator(all_data)
            # Calculate loss between output of discriminator and expected labels
            loss_disc = loss_function(output_disc, all_labels)
            # print(loss_disc)
            # Back propagate with the loss and step the discriminator
            loss_disc.backward()
            optimiser_discriminator.step()

            # Samples of latent data to train Generator
            # Latent data sample for diff
            latent_data_samples = torch.randint(
                0, diff_values, (sequence_length, batch_size)
            )
            # Latent data sample for length
            # latent_data_samples = torch.randint(0, len_values, (sequence_length, batch_size))

            # Train generator
            generator.zero_grad()
            # Generate data with Generator
            generated_data = generator(latent_data_samples)
            # Get discriminator predictions of data being real
            output_disc_gen = discriminator(generated_data)

            # Calculate loss based on difference between correctly assigned
            loss_gen = loss_function(output_disc_gen, real_data_labels)
            loss_gen.backward()
            # print(loss_gen)
            optimiser_generator.step()

            num_trained_batches += 1

            # Save performance and output results while running
            if num_trained_batches % 50 == 0:
                print(f"Batches:  {num_trained_batches}/{len(data_loader)}")
                print(f"Discriminator Loss: {loss_disc}")
                print(f"Generator Loss: {loss_gen}")
                generator_loss.append(loss_gen.item())
                discriminator_loss.append(loss_disc.item())
                batch_history.append(num_trained_batches)

    return generator, discriminator, generator_loss, discriminator_loss, batch_history


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
    files_to_parse = 4
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
        num_heads,
        num_layers,
        dropout=0.0,
    ).to(Run_Transformer_Model.getDevice())
    # length generator
    # generator = Generator(sequence_length, len_values, num_heads * embedded_heads, num_heads, num_layers, dropout=0.0).to(Run_Transformer_Model.getDevice())

    loss_function = nn.BCEWithLogitsLoss()
    optimiser_generator = torch.optim.Adam(generator.parameters(), lr=0.001)

    # Build discriminator and define optimisation method
    discriminator = GAN_Transformer_Pitch.Discriminator(batch_size, sequence_length).to(
        Run_Transformer_Model.getDevice()
    )
    optimiser_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0000001)

    # Set both models to train
    generator.train()
    discriminator.train()

    (
        generator,
        discriminator,
        generator_loss,
        discriminator_loss,
        batch_history,
    ) = trainModel(
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
    #     os.path.join(os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Diff.pth"),
    # )

    # Save Len Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Len.pth"
    #     ),
    # )

    # Save Diff Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Discriminator_Model_Diff.pth"
    #     ),
    # )

    # Save Len Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Discriminator_Model_Len.pth"
    #     ),
    # )

    # Save diff Generator
    torch.save(
        generator.state_dict(),
        os.path.join(os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Diff_Electronic.pth"),
    )

    # Save Len Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Len_Electronic.pth"
    #     ),
    # )

    # Save Diff Discriminator
    torch.save(
        discriminator.state_dict(),
        os.path.join(
            os.sys.path[0], "GAN_Transformer/GAN_Discriminator_Model_Diff_Electronic.pth"
        ),
    )

    # Save Len Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Discriminator_Model_Len_Electronic.pth"
    #     ),
    # )

    # Save diff Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Diff_Jazz.pth"),
    # )

    # Save Len Generator
    # torch.save(
    #     generator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Transformer_Model_Len_Jazz.pth"
    #     ),
    # )

    # Save Diff Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Discriminator_Model_Diff_Jazz.pth"
    #     ),
    # )

    # Save Len Discriminator
    # torch.save(
    #     discriminator.state_dict(),
    #     os.path.join(
    #         os.sys.path[0], "GAN_Transformer/GAN_Discriminator_Model_Len_Jazz.pth"
    #     ),
    # )

    print("Model saved")
