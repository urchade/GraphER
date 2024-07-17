import torch
import torch.nn.functional as F
from torch import nn

from .layers import MLP


class ScorerLayer(nn.Module):
    def __init__(self, scoring_type="dot", hidden_size=768, dropout=0.1):
        super().__init__()

        self.scoring_type = scoring_type

        if scoring_type == "concat_proj":
            self.proj = MLP([hidden_size * 4, hidden_size * 4, 1], dropout)
        elif scoring_type == "dot_thresh":
            self.proj_thresh = MLP([hidden_size, hidden_size * 4, 2], dropout)
            self.proj_type = MLP([hidden_size, hidden_size * 4, hidden_size], dropout)

    def forward(self, candidate_pair_rep, rel_type_rep):
        # candidate_pair_rep: [B, N, D]
        # rel_type_rep: [B, T, D]
        if self.scoring_type == "dot":
            return torch.einsum("bnd,btd->bnt", candidate_pair_rep, rel_type_rep)

        elif self.scoring_type == "dot_thresh":
            # compute the scaling factor and threshold
            B, T, D = rel_type_rep.size()
            scaler = self.proj_thresh(rel_type_rep)  # [B, T, 2]
            # alpha: scaling factor, beta: threshold
            alpha, beta = scaler[..., 0].view(B, 1, T), scaler[..., 1].view(B, 1, T)
            alpha = F.softplus(alpha)  # reason: alpha should be positive
            # project the relation type representation
            rel_type_rep = self.proj_type(rel_type_rep)  # [B, T, D]
            # compute the score (before sigmoid)
            score = torch.einsum("bnd,btd->bnt", candidate_pair_rep, rel_type_rep)  # [B, N, T]
            return (score + beta) * alpha  # [B, N, T]

        elif self.scoring_type == "dot_norm":
            score = torch.einsum("bnd,btd->bnt", candidate_pair_rep, rel_type_rep)  # [B, N, T]
            bias_1 = self.dy_bias_type(rel_type_rep).transpose(1, 2)  # [B, 1, T]
            bias_2 = self.dy_bias_rel(candidate_pair_rep)  # [B, N, 1]
            return score + self.bias + bias_1 + bias_2

        elif self.scoring_type == "concat_proj":
            prod_features = candidate_pair_rep.unsqueeze(2) * rel_type_rep.unsqueeze(1)  # [B, N, T, D]
            diff_features = candidate_pair_rep.unsqueeze(2) - rel_type_rep.unsqueeze(1)  # [B, N, T, D]
            expanded_pair_rep = candidate_pair_rep.unsqueeze(2).repeat(1, 1, rel_type_rep.size(1), 1)
            expanded_rel_type_rep = rel_type_rep.unsqueeze(1).repeat(1, candidate_pair_rep.size(1), 1, 1)
            features = torch.cat([prod_features, diff_features, expanded_pair_rep, expanded_rel_type_rep],
                                 dim=-1)  # [B, N, T, 2D]
            return self.proj(features).squeeze(-1)
