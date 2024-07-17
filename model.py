import torch
import torch.nn.functional as F
from torch import nn

from modules.base import GrapherBase
from modules.data_processor import TokenPromptProcessorTR
from modules.filtering import FilteringLayer
from modules.layers import MLP, LstmSeq2SeqEncoder, TransLayer, GraphEmbedder
from modules.loss_functions import compute_matching_loss
from modules.rel_rep import RelationRep
from modules.scorer import ScorerLayer
from modules.span_rep import SpanRepLayer
from modules.token_rep import TokenRepLayer
from modules.utils import get_ground_truth_relations, get_candidates


class GraphER(GrapherBase):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # [ENT] token
        self.ent_token = "<<ENT>>"
        self.rel_token = "<<REL>>"
        self.sep_token = "<<SEP>>"

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling, hidden_size=config.hidden_size,
                                             add_tokens=[self.ent_token, self.rel_token, self.sep_token])

        # token prompt processor
        self.token_prompt_processor = TokenPromptProcessorTR(self.ent_token, self.rel_token, self.sep_token)

        # hierarchical representation of tokens (Zaratiana et al, 2022)
        # https://arxiv.org/pdf/2203.14710.pdf
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True
        )

        # span representation
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout
        )

        # prompt representation (FFN)
        self.ent_rep_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.rel_rep_layer = nn.Linear(config.hidden_size, config.hidden_size)

        # filtering layer for spans and relations
        self._span_filtering = FilteringLayer(config.hidden_size)
        self._rel_filtering = FilteringLayer(config.hidden_size)

        # relation representation
        self.relation_rep = RelationRep(config.hidden_size, config.dropout, config.ffn_mul)

        # graph embedder
        self.graph_embedder = GraphEmbedder(config.hidden_size)

        # transformer layer
        self.trans_layer = TransLayer(
            config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers
        )

        # keep_mlp
        self.keep_mlp = MLP([config.hidden_size, config.hidden_size * config.ffn_mul, 1], dropout=0.1)

        # scoring layers
        self.scorer_ent = ScorerLayer(
            scoring_type=config.scorer,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

        self.scorer_rel = ScorerLayer(
            scoring_type=config.scorer,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

    def get_optimizer(self, lr_encoder, lr_others, freeze_token_rep=False):
        """
        Parameters:
        - lr_encoder: Learning rate for the encoder layer.
        - lr_others: Learning rate for all other layers.
        - freeze_token_rep: whether the token representation layer should be frozen.
        """
        param_groups = [
            # encoder
            {"params": self.rnn.parameters(), "lr": lr_others},
            # projection layers
            {"params": self.span_rep_layer.parameters(), "lr": lr_others},
            # prompt representation
            {"params": self.ent_rep_layer.parameters(), "lr": lr_others},
            {"params": self.rel_rep_layer.parameters(), "lr": lr_others},
            # filtering layers
            {"params": self._span_filtering.parameters(), "lr": lr_others},
            {"params": self._rel_filtering.parameters(), "lr": lr_others},
            # relation representation
            {"params": self.relation_rep.parameters(), "lr": lr_others},
            # graph embedder
            {"params": self.graph_embedder.parameters(), "lr": lr_others},
            # transformer layer
            {"params": self.trans_layer.parameters(), "lr": lr_others},
            # keep_mlp
            {"params": self.keep_mlp.parameters(), "lr": lr_others},
            # scoring layer
            {"params": self.scorer_ent.parameters(), "lr": lr_others},
            {"params": self.scorer_rel.parameters(), "lr": lr_others}
        ]

        if not freeze_token_rep:
            # If token_rep_layer should not be frozen, add it to the optimizer with its learning rate
            param_groups.append({"params": self.token_rep_layer.parameters(), "lr": lr_encoder})
        else:
            # If token_rep_layer should be frozen, explicitly set requires_grad to False for its parameters
            for param in self.token_rep_layer.parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(param_groups)

        return optimizer

    def compute_score_train(self, x):
        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)

        # Process input
        word_rep, mask, entity_type_rep, entity_type_mask, rel_type_rep, relation_type_mask = self.token_prompt_processor.process(
            x, self.token_rep_layer, "train"
        )

        # Compute representations
        word_rep = self.rnn(word_rep, mask)
        span_rep = self.span_rep_layer(word_rep, span_idx)
        entity_type_rep = self.ent_rep_layer(entity_type_rep)
        rel_type_rep = self.rel_rep_layer(rel_type_rep)

        # Compute number of entity and relation types
        num_ent, num_rel = entity_type_rep.shape[1], rel_type_rep.shape[1]

        return span_rep, num_ent, num_rel, entity_type_rep, entity_type_mask, rel_type_rep, relation_type_mask, (
            word_rep, mask)

    @torch.no_grad()
    def compute_score_eval(self, x, device):
        span_idx = (x['span_idx'] * x['span_mask'].unsqueeze(-1)).to(device)

        # Process input
        word_rep, mask, entity_type_rep, relation_type_rep = self.token_prompt_processor.process(
            x, self.token_rep_layer, "eval"
        )

        # Compute representations
        word_rep = self.rnn(word_rep, mask)
        span_rep = self.span_rep_layer(word_rep, span_idx)
        entity_type_rep = self.ent_rep_layer(entity_type_rep)
        relation_type_rep = self.rel_rep_layer(relation_type_rep)

        # Compute number of entity and relation types
        num_ent, num_rel = entity_type_rep.shape[1], relation_type_rep.shape[1]

        return span_rep, num_ent, num_rel, entity_type_rep, relation_type_rep, (word_rep, mask)

    def forward(self, x, prediction_mode=False):

        # clone span_label
        span_label = x['span_label'].clone()

        # compute span representation
        if prediction_mode:
            # Get the device of the model parameters
            device = next(self.parameters()).device

            # Compute scores for evaluation
            span_rep, num_ent, num_rel, entity_type_rep, rel_type_rep, (word_rep, word_mask) = self.compute_score_eval(
                x, device)

            # Create masks for relation and entity types, setting all values to 1
            relation_type_mask = torch.ones(size=(rel_type_rep.shape[0], num_rel), device=device)
            entity_type_mask = torch.ones(size=(entity_type_rep.shape[0], num_ent), device=device)
        else:
            # Compute scores for training
            span_rep, num_ent, num_rel, entity_type_rep, entity_type_mask, rel_type_rep, relation_type_mask, (
            word_rep, mask) = self.compute_score_train(x)

        # Reshape span_rep from (B, L, K, D) to (B, L * K, D)
        B, L, K, D = span_rep.shape
        span_rep = span_rep.view(B, L * K, D)

        # Compute filtering scores for spans
        filter_score_span, filter_loss_span = self._span_filtering(span_rep, x['span_label'])

        # Determine the maximum number of candidates
        # If L is greater than the configured maximum, use the configured maximum plus an additional top K
        # Otherwise, use L plus an additional top K
        max_top_k = min(L, self.config.max_top_k) + self.config.add_top_k

        # Sort the filter scores for spans in descending order
        sorted_idx = torch.sort(filter_score_span, dim=-1, descending=True)[1]

        # Define the elements to get candidates for
        elements = [span_rep, span_label, x['span_mask'], x['span_idx']]

        # Use a list comprehension to get the candidates for each element
        candidate_span_rep, candidate_span_label, candidate_span_mask, candidate_spans_idx = [
            get_candidates(sorted_idx, element, topk=max_top_k)[0] for element in elements
        ]

        # Calculate the lengths for the top K entities
        top_k_lengths = x["seq_length"].clone() + self.config.add_top_k

        # Create a condition mask where the range of top K is greater than or equal to the top K lengths
        condition_mask = torch.arange(max_top_k, device=span_rep.device).unsqueeze(0) >= top_k_lengths.unsqueeze(-1)

        # Apply the condition mask to the candidate span mask and label, setting the masked values to 0 and -1
        # respectively
        candidate_span_mask.masked_fill_(condition_mask, 0)
        candidate_span_label.masked_fill_(condition_mask, -1)

        # Get ground truth relations and represent them
        relation_classes = get_ground_truth_relations(x, candidate_spans_idx, candidate_span_label)
        rel_rep = self.relation_rep(candidate_span_rep).view(B, max_top_k * max_top_k, -1)  # Reshape in the same line

        # Compute filtering scores for relations and sort them in descending order
        filter_score_rel, filter_loss_rel = self._rel_filtering(rel_rep, relation_classes)
        sorted_idx_pair = torch.sort(filter_score_rel, dim=-1, descending=True)[1]

        # Embed candidate span representations
        candidate_span_rep, cat_pair_rep = self.graph_embedder(candidate_span_rep)

        # Define the elements to get candidates for
        elements = [cat_pair_rep.view(B, max_top_k * max_top_k, -1), relation_classes.view(B, max_top_k * max_top_k)]

        # Use a list comprehension to get the candidates for each element
        candidate_pair_rep, candidate_pair_label = [get_candidates(sorted_idx_pair, element, topk=max_top_k)[0] for
                                                    element in elements]

        # Get the top K relation indices
        topK_rel_idx = sorted_idx_pair[:, :max_top_k]

        # Mask the candidate pair labels using the condition mask and refine the relation representation
        candidate_pair_label.masked_fill_(condition_mask, -1)
        candidate_pair_mask = candidate_pair_label > -1

        # Concatenate span and relation representations
        concat_span_pair = torch.cat((candidate_span_rep, candidate_pair_rep), dim=1)
        mask_span_pair = torch.cat((candidate_span_mask, candidate_pair_mask), dim=1)

        # Apply transformer layer and keep_mlp
        out_trans = self.trans_layer(concat_span_pair, mask_span_pair)
        keep_score = self.keep_mlp(out_trans).squeeze(-1)  # Shape: (B, max_top_k + max_top_k, 1)

        # Apply sigmoid function and squeeze the last dimension
        # keep_score = torch.sigmoid(keep_score).squeeze(-1)  # Shape: (B, max_top_k + max_top_k)

        # Split keep_score into keep_ent and keep_rel
        keep_ent, keep_rel = keep_score.split([max_top_k, max_top_k], dim=1)

        """not use output from transformer layer for now"""
        # Split out_trans
        # candidate_span_rep, candidate_pair_rep = out_trans.split([max_top_k, max_top_k], dim=1)

        # Compute scores for entities and relations
        scores_ent = self.scorer_ent(candidate_span_rep, entity_type_rep)  # Shape: [B, N, C]
        scores_rel = self.scorer_rel(candidate_pair_rep, rel_type_rep)  # Shape: [B, N, C]

        if prediction_mode:
            return {
                "entity_logits": scores_ent,
                "relation_logits": scores_rel,
                "keep_ent": keep_ent,
                "keep_rel": keep_rel,
                "candidate_spans_idx": candidate_spans_idx,
                "candidate_pair_label": candidate_pair_label,
                "max_top_k": max_top_k,
                "topK_rel_idx": topK_rel_idx
            }
        # Compute losses for relation and entity classifiers
        relation_loss = compute_matching_loss(scores_rel, candidate_pair_label, relation_type_mask, num_rel)
        entity_loss = compute_matching_loss(scores_ent, candidate_span_label, entity_type_mask, num_ent)

        # Concatenate labels for binary classification and compute binary classification loss
        ent_rel_label = (torch.cat((candidate_span_label, candidate_pair_label), dim=1) > 0).float()
        filter_loss = F.binary_cross_entropy_with_logits(keep_score, ent_rel_label, reduction='none')

        # Compute structure loss and total loss
        structure_loss = (filter_loss * mask_span_pair.float()).sum()
        total_loss = sum([filter_loss_span, filter_loss_rel, relation_loss, entity_loss, structure_loss])

        return total_loss
