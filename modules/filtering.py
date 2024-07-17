from torch import nn
from .loss_functions import down_weight_loss

class FilteringLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.filter_layer = nn.Linear(hidden_size, 2)

    def forward(self, embeds, label):

        # Extract dimensions
        B, num_spans, D = embeds.shape

        # Compute score using a predefined filtering function
        score = self.filter_layer(embeds)  # Shape: [B, num_spans, num_classes]

        # Modify label to binary (0 for negative class, 1 for positive)
        label_m = label.clone()
        label_m[label_m > 0] = 1

        # Initialize the loss
        filter_loss = 0
        if self.training:
            # Compute the loss if in training mode
            filter_loss = down_weight_loss(score.view(B * num_spans, -1),
                                           label_m.view(-1),
                                           sample_rate=0.,
                                           is_logit=True)

        # Compute the filter score (difference between positive and negative class scores)
        filter_score = score[..., 1] - score[..., 0]  # Shape: [B, num_spans]

        # Mask out filter scores for ignored labels
        filter_score = filter_score.masked_fill(label == -1, float('-inf'))

        if self.training:
            filter_score = filter_score.masked_fill(label_m > 0, float('inf'))

        return filter_score, filter_loss
