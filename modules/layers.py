import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def MLP(units, dropout, activation=nn.ReLU):
    units = [int(u) for u in units]
    assert len(units) >= 2
    layers = []
    for i in range(len(units) - 2):
        layers.append(nn.Linear(units[i], units[i + 1]))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(units[-2], units[-1]))
    return nn.Sequential(*layers)


def create_transformer_encoder(d_model, nhead, num_layers, ffn_mul=4, dropout=0.1):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, batch_first=True, norm_first=False, dim_feedforward=d_model * ffn_mul,
        dropout=dropout)
    encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    return encoder


class TransLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, ffn_mul=4, dropout=0.1):
        super(TransLayer, self).__init__()

        if num_layers > 0:
            self.transformer_encoder = create_transformer_encoder(d_model, num_heads, num_layers, ffn_mul, dropout)

    def forward(self, x, mask):
        mask = mask == False
        if not hasattr(self, 'transformer_encoder'):
            return x
        else:
            return self.transformer_encoder(src=x, src_key_padding_mask=mask)


class GraphEmbedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # Project node to half of its dimension
        self.project_node = nn.Linear(d_model, d_model // 2)

        # Initialize identifier with zeros
        self.identifier = nn.Parameter(torch.randn(2, d_model))
        nn.init.zeros_(self.identifier)

    def forward(self, candidate_span_rep):
        max_top_k = candidate_span_rep.size()[1]

        # Project nodes
        nodes = self.project_node(candidate_span_rep)

        # Split nodes into heads and tails
        heads = nodes.unsqueeze(2).expand(-1, -1, max_top_k, -1)
        tails = nodes.unsqueeze(1).expand(-1, max_top_k, -1, -1)

        # Concatenate heads and tails to form edges
        edges = torch.cat([heads, tails], dim=-1)

        # Duplicate nodes along the last dimension
        nodes = torch.cat([nodes, nodes], dim=-1)

        # Add identifier to nodes and edges
        nodes += self.identifier[0]
        edges += self.identifier[1]

        return nodes, edges


class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False):
        super(LstmSeq2SeqEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)

        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output
