import math
import numpy as np
import torch
import torch.nn as nn
from utils import *
import Run_Transformer_Model
import Timings_Multihead_Attention

# Transformer class for the Timings
class TimingsTransformer(nn.Module):
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
                Timings_Multihead_Attention.TimingsMultiheadAttention(
                    num_inputs, sequence_length, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    # Method defines how to proceed for each layer
    def forward(self, input: np.array):
        # Encode input
        input = self.encoding(input) * math.sqrt(self.num_inputs)

        # Add positional embeddings for the input
        input = torch.cat([input, self.pos_embeds], axis=2)

        # Run over the layers (heads) in the multi
        for layer in range(self.num_layers):
            input = input.swapaxes(0, 1)
            # Normalise the batches
            input = self.batch_norm[layer](input)
            input = input.swapaxes(0, 1)
            # Run attention layers
            input = self.transformer_attention[layer](input)
            input = input.swapaxes(0, 1)

        # Decode the output
        output = self.decoder(input)

        return output