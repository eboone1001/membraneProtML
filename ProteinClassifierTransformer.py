# This is the implementation of a transformer classifier to solve the same problem posed as Protein Classifier
# It should be significantly faster, allowing for a larger training base, thus better performance.
# Author: Eric Boone
# Date: 11/2/2022

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import membraneProtML_helper as hlpr
import math


class ProtSeqEncoder(nn.Module):
    """
    This constructs and pools the positional encoding matrix.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MemProtClassTransformer(nn.Module):

    def __init__(self, ntoken: int, nclasses: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.model_type = "Transformer"
        self.pos_encoder = ProtSeqEncoder(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        # TODO: output is coming out as 3D matrix; don't think this is right, figure out
        # TODO: Convert into classifier; add linear layer with output vector size num_classes

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def train(model: nn.Module, filename) -> None:
    model.train()
    data, targets = hlpr.read_data(filename)
    targets = hlpr.int_labels_to_onehot(targets)

    for i in range(len(data)):
        seq = data[i]
        targ = torch.tensor(targets[i])

        seq_len = len(seq)
        src_mask = generate_square_subsequent_mask(seq_len)

        seq_tensor = hlpr.sequence_to_tensor(seq, ntokens)

        output = model(seq_tensor, src_mask)
        loss = criterion(output.view(-1, ntokens), targ)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


def evaluate(model: nn.Module, filename) -> float:
    model.eval()  # turn on evaluation mod
    eval_data, targets = hlpr.read_data(filename)

    total_loss = 0.
    with torch.no_grad():
        for seq in eval_data:
            seq_len = len(seq)
            src_mask = generate_square_subsequent_mask(seq_len)

            output = model(seq, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


if __name__ == "__main__":
    # TODO: Model just uses these varables, keep them here and add the missing ones to parameters
    # TODO: Add command line interface so this can be trained on a schools datahub GPUs
    # Model hyperparameters
    ntokens = 26  # Number of AA symbols
    nclasses = 2
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.1  # dropout probability
    lr = 5.0  # learning rate

    model = MemProtClassTransformer(ntokens, nclasses, emsize, nhead, d_hid, nlayers, dropout)


    # Pytorch training functions
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # TODO: find out if having a train() and eval() routine is best practice.
    train(model, "rawdata/data_train.csv")
    evaluate(model, "rawdata/data_test.csv")
