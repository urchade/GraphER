import torch
from torch import nn
from .layers import MLP


class RelationRep(nn.Module):
    def __init__(self, hidden_size, dropout, ffn_mul):
        super().__init__()

        self.head_mlp = nn.Linear(hidden_size, hidden_size // 2)
        self.tail_mlp = nn.Linear(hidden_size, hidden_size // 2)
        self.out_mlp = MLP([hidden_size, hidden_size * ffn_mul, hidden_size], dropout)

    def forward(self, span_reps):
        """
        :param span_reps [B, topk, D]
        :return relation_reps [B, topk, topk, D]
        """

        heads, tails = span_reps, span_reps

        # Apply MLPs to heads and tails
        heads = self.head_mlp(heads)
        tails = self.tail_mlp(tails)

        # Expand heads and tails to create relation representations
        heads = heads.unsqueeze(2).expand(-1, -1, heads.shape[1], -1)
        tails = tails.unsqueeze(1).expand(-1, tails.shape[1], -1, -1)

        # Concatenate heads and tails to create relation representations
        relation_reps = torch.cat([heads, tails], dim=-1)

        # Apply MLP to relation representations
        relation_reps = self.out_mlp(relation_reps)

        return relation_reps
