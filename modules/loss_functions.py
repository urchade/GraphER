import torch
import torch.nn.functional as F


def compute_loss(logits, labels, mask, num_classes):
    B, _, _ = logits.size()

    logits_label = logits.view(-1, num_classes)
    labels = labels.view(-1)  # (batch_size * num_spans)
    mask_label = labels != -1  # (batch_size * num_spans)
    labels.masked_fill_(~mask_label, 0)  # Set the labels of padding tokens to 0

    # one-hot encoding
    labels_one_hot = torch.zeros(labels.size(0), num_classes + 1, dtype=torch.float32).to(logits.device)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)  # Set the corresponding index to 1
    labels_one_hot = labels_one_hot[:, 1:]  # Remove the first column

    # loss for classifier
    loss = F.binary_cross_entropy_with_logits(logits_label, labels_one_hot, reduction='none')
    # mask loss using mask (B, C)
    masked_loss = loss.view(B, -1, num_classes) * mask.unsqueeze(1)
    loss = masked_loss.view(-1, num_classes)
    # expand mask_label to loss
    mask_label = mask_label.unsqueeze(-1).expand_as(loss)
    # put lower loss for in labels_one_hot (2 for positive, 1 for negative)

    # apply mask
    loss = loss * mask_label.float()
    loss = loss.sum()

    return loss


def down_weight_loss(logits, y, sample_rate=0.1, is_logit=True):

    if is_logit:
        loss_func = F.cross_entropy
    else:
        loss_func = F.nll_loss

    loss_entity = loss_func(logits, y.masked_fill(y == 0, -1), ignore_index=-1, reduction='sum')
    loss_non_entity = loss_func(logits, y.masked_fill(y > 0, -1), ignore_index=-1, reduction='sum')

    return loss_entity + loss_non_entity * (1 - sample_rate)
