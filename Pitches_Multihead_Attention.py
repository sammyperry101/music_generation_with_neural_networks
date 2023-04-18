import numpy as np
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
import Run_Transformer_Model

# Class which controls the multi-head attention for the note pitches
class PitchesMultiheadAttention(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        sequence_length: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initate class variables
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        # Number of features per head
        feat_per_head = embedding_size // num_heads

        # Store information on each head
        self.head_information = []
        for _ in range(self.num_heads):
            self.head_information.append(
                torch.nn.ModuleList(
                    [
                        nn.Linear(
                            in_features=feat_per_head,
                            out_features=feat_per_head,
                            bias=False,
                        )
                        for _ in range(3)
                    ]
                )
            )

        # Combine heads together to create output of the correct size
        self.combined_heads = nn.Linear(
            in_features=num_heads * feat_per_head, out_features=embedding_size
        )

        # Define the dropout layer
        self.dropout = nn.Dropout(p=dropout)
        self.relative_pos_embedding = torch.randn(
            [feat_per_head, sequence_length],
            device=Run_Transformer_Model.getDevice(),
            requires_grad=True,
        )

    # Method defines how to proceed for each layer
    def forward(self, x: np.array):
        # Permute input and return size of x
        x = x.permute(1, 0, 2)
        batch_size, sequence_len, embedding_dimension = x.size()
        single_head = embedding_dimension // self.num_heads
        # Divide the input into different dimesions for each head
        x = x.view(batch_size, sequence_len, self.num_heads, single_head)

        # Calculate results for seperate queries, keys and values for each head used
        queries = []
        keys = []
        values = []
        for head in range(self.num_heads):
            x_head = x[:, :, head, :]
            query, key, value = [
                w(x)
                for w, x in zip(
                    self.head_information[head].to(Run_Transformer_Model.getDevice()),
                    (x_head, x_head, x_head),
                )
            ]
            
            # Append results
            queries.append(query)
            keys.append(key)
            values.append(value)

        # Apply consistent positional embeddings across model
        relative_attention = []
        for head in range(self.num_heads):
            queries_pos_embedding = torch.matmul(
                queries[head], self.relative_pos_embedding
            )
            relative_attention.append(
                queries_pos_embedding.contiguous().view(
                    batch_size, sequence_len, sequence_len
                )
            )

        # Compute self attention
        head_rep = []
        for head in range(self.num_heads):
            # Scale multiplication to allow for stability
            queries[head] = queries[head] / (embedding_dimension ** (1 / 4))
            keys[head] = keys[head] / (embedding_dimension ** (1 / 4))

            # Multiply keys and queries together
            scores_head = torch.bmm(queries[head], keys[head].transpose(1, 2))

            # Add attention score to relative positional score
            scores = scores_head + relative_attention[head]

            # Create Mask
            head_mask = torch.triu(
                torch.ones(
                    1,
                    sequence_len,
                    sequence_len,
                    device=Run_Transformer_Model.getDevice(),
                ),
                1,
            )
            scores = scores.masked_fill(head_mask == 1, -2e8)

            # Convert scores to probabilities
            attn_probs = F.softmax(scores, dim=2)
            attn_probs = self.dropout(attn_probs)

            # Get weighted average of values using the attention
            head_rep.append(
                torch.bmm(attn_probs, values[head]).view(
                    batch_size, sequence_len, single_head
                )
            )

        # Recombine and transpose the heads
        out = torch.cat(head_rep, dim=2)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_len, single_head * self.num_heads)
        )

        # Return the combined heads
        return self.combined_heads(out)