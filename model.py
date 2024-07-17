import torch
import torch.nn.functional as F
from torch import nn

from modules.base import GrapherBase
from modules.data_processor import TokenPromptProcessorTR
from modules.filtering import FilteringLayer
from modules.layers import TransLayer, GraphEmbedder, MLP, LstmSeq2SeqEncoder
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
                                             add_tokens=[self.rel_token, self.sep_token])

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
        self.trans_layer = TransLayer(config.hidden_size, num_heads=4, num_layers=2)

        # keep_mlp
        self.keep_mlp = MLP([config.hidden_size, config.hidden_size * config.ffn_mul, 1], dropout=0.1)

        # scoring layer
        self.scorer_ent = ScorerLayer(scoring_type=config.scorer, hidden_size=config.hidden_size,
                                      dropout=config.dropout)
        self.scorer_rel = ScorerLayer(scoring_type=config.scorer, hidden_size=config.hidden_size,
                                      dropout=config.dropout)

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
            device = next(self.parameters()).device
            # span_rep, num_ent, num_rel, entity_type_rep, relation_type_rep, None, (word_rep, mask)
            span_rep, num_ent, num_rel, entity_type_rep, rel_type_rep, (word_rep, word_mask) = self.compute_score_eval(
                x,
                device)
            # set relation_type_mask to tensor of ones
            relation_type_mask = torch.ones(rel_type_rep.shape[0], num_rel).to(device)

            # set entity_type_mask to tensor of ones
            entity_type_mask = torch.ones(entity_type_rep.shape[0], num_ent).to(device)
        else:
            span_rep, num_ent, num_rel, entity_type_rep, entity_type_mask, rel_type_rep, relation_type_mask, (
                word_rep, mask) = self.compute_score_train(x)

        B, L, K, D = span_rep.shape
        span_rep = span_rep.view(B, L * K, D)

        # filtering scores for spans
        filter_score_span, filter_loss_span = self._span_filtering(span_rep, x['span_label'])

        # number of candidates
        max_top_k = L + self.config.add_top_k

        if L > self.config.max_top_k:
            max_top_k = self.config.max_top_k + self.config.add_top_k

        # filtering scores for spans
        _, sorted_idx = torch.sort(filter_score_span, dim=-1, descending=True)

        # Get candidate spans and labels
        candidate_span_rep, candidate_span_label, candidate_span_mask, candidate_spans_idx = [
            get_candidates(sorted_idx, el, topk=max_top_k)[0] for el in
            [span_rep, span_label, x['span_mask'], x['span_idx']]]

        # configure masks for entity #############################################
        ##########################################################################
        top_k_lengths = x["seq_length"].clone() + self.config.add_top_k
        arange_topk = torch.arange(max_top_k, device=span_rep.device)
        masked_fill_cond = arange_topk.unsqueeze(0) >= top_k_lengths.unsqueeze(-1)
        candidate_span_mask.masked_fill_(masked_fill_cond, 0)
        candidate_span_label.masked_fill_(masked_fill_cond, -1)
        ##########################################################################
        ##########################################################################

        # get ground truth relations
        relation_classes = get_ground_truth_relations(x, candidate_spans_idx, candidate_span_label)

        # representation of relations
        rel_rep = self.relation_rep(candidate_span_rep)  # [B, topk, topk, D]

        # filtering scores for relations
        filter_score_rel, filter_loss_rel = self._rel_filtering(
            rel_rep.view(B, max_top_k * max_top_k, -1), relation_classes)

        # filtering scores for relation pairs
        _, sorted_idx_pair = torch.sort(filter_score_rel, dim=-1, descending=True)

        candidate_span_rep, cat_pair_rep = self.graph_embedder(candidate_span_rep)

        # Get candidate pairs and labels
        candidate_pair_rep, candidate_pair_label = [get_candidates(sorted_idx_pair, el, topk=max_top_k)[0] for el
                                                    in
                                                    [cat_pair_rep.view(B, max_top_k * max_top_k, -1),
                                                     relation_classes.view(B, max_top_k * max_top_k)]]

        topK_rel_idx = sorted_idx_pair[:, :max_top_k]

        #######################################################
        candidate_pair_label.masked_fill_(masked_fill_cond, -1)
        #######################################################

        # refine relation representation ##############################################
        candidate_pair_mask = candidate_pair_label > -1
        ################################################################################

        # concat span and relation representation
        # outcont of shape (B, max_top_k + max_top_k, D) # ent1, ent2, ..., ent_n, rel1, rel2, ..., rel_n
        out_cont = torch.cat((candidate_span_rep, candidate_pair_rep), dim=1)

        # mask for relation type representation
        mask_cont = torch.cat((candidate_span_mask, candidate_pair_mask),
                              dim=1)

        # transformer layer
        out_trans = self.trans_layer(out_cont, mask_cont)

        # keep_mlp
        keep_score = self.keep_mlp(out_trans)  # (B, max_top_k + max_top_k, 2)

        # keep_score[..., 0] is for ent, keep_score[..., 1] is for rel
        keep_score_ent = keep_score[:, :max_top_k, 0].unsqueeze(-1)  # (B, max_top_k, 1)
        keep_score_rel = keep_score[:, max_top_k:, 0].unsqueeze(-1)  # (B, max_top_k, 1)

        keep_score = torch.cat((keep_score_ent, keep_score_rel), dim=1)  # (B, max_top_k + max_top_k, 1)

        keep_score = torch.sigmoid(keep_score).squeeze(-1)  # (B, max_top_k + max_top_k)

        keep_ent, keep_rel = keep_score.split([max_top_k, max_top_k], dim=1)

        # compute scores
        scores_ent = self.scorer_ent(candidate_span_rep, entity_type_rep)  # [B, N, C]
        scores_rel = self.scorer_rel(candidate_pair_rep, rel_type_rep)  # [B, N, C]

        if prediction_mode:
            return {"entity_logits": scores_ent, "relation_logits": scores_rel,
                    "candidate_spans_idx": candidate_spans_idx,
                    "candidate_pair_label": candidate_pair_label,
                    "max_top_k": max_top_k, "topK_rel_idx": topK_rel_idx, "keep_ent": keep_ent, "keep_rel": keep_rel}

        # loss for relation classifier
        relation_loss = self.compute_loss(scores_rel, candidate_pair_label, relation_type_mask, num_rel)
        entity_loss = self.compute_loss(scores_ent, candidate_span_label, entity_type_mask, num_ent)

        # concat label for binary classification
        ent_rel_label = torch.cat((candidate_span_label, candidate_pair_label), dim=1) > 0  # (B, max_top_k + max_top_k)

        # binary classification loss
        filter_loss = F.binary_cross_entropy(keep_score, ent_rel_label.float(), reduction='none')

        structure_loss = (filter_loss * mask_cont.float()).sum()

        return filter_loss_span + filter_loss_rel + relation_loss + entity_loss + structure_loss
