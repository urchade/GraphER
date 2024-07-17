from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random


# Abstract base class for handling data processing
class GrapherData(object):
    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_dict(spans: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        """Get a dictionary of spans."""
        dict_tag = defaultdict(int)
        for span in spans:
            dict_tag[(span[0], span[1])] = classes_to_id[span[-1]]
        return dict_tag

    def preprocess_spans(self, tokens: List[str], ner: List[Tuple[int, int, str]], rel: List[Tuple[int, int, str]],
                         classes_to_id: Dict[str, int]) -> Dict:
        """Preprocess spans for a given text."""
        # Set the maximum length for tokens
        max_token_length = self.config.max_len

        # If the number of tokens exceeds the maximum length, truncate the tokens
        if len(tokens) > max_token_length:
            token_length = max_token_length
            tokens = tokens[:max_token_length]
        else:
            token_length = len(tokens)

        # Initialize a list to store span indices
        span_indices = []
        for i in range(token_length):
            span_indices.extend([(i, i + j) for j in range(self.config.max_width)])

        # Get the dictionary of labels
        label_dict = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)

        # Initialize the span labels with the corresponding label from the dictionary
        span_labels = torch.LongTensor([label_dict[i] for i in span_indices])
        span_indices = torch.LongTensor(span_indices)

        # Create a mask for valid spans
        valid_span_mask = span_indices[:, 1] > token_length - 1

        # Mask invalid positions in the span labels
        span_labels = span_labels.masked_fill(valid_span_mask, -1)

        # Return a dictionary with the preprocessed spans
        return {
            'tokens': tokens,
            'span_idx': span_indices,
            'span_label': span_labels,
            'seq_length': token_length,
            'entities': ner,
            'relations': rel,
        }

    def create_mapping(self, types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create a mapping from type to id and id to type."""
        if not types:
            types = ["None"]
        type_to_id = {type: id for id, type in enumerate(types, start=1)}
        id_to_type = {id: type for type, id in type_to_id.items()}
        return type_to_id, id_to_type

    def batch_generate_class_mappings(self, batch_list: List[Dict]) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]], List[Dict[str, int]], List[Dict[int, str]]]:
        """Generate class mappings for a batch of data."""
        all_ent_to_id, all_id_to_ent, all_rel_to_id, all_id_to_rel = [], [], [], []

        negative_entities = []
        negative_relations = []

        for b in batch_list:
            ent_types = list(set([el[-1] for el in b['entities']]))
            rel_types = list(set([el[-1] for el in b['relations']]))

            # added hpm: random_drop, max_neg_type_ratio
            # max_ent_types, max_rel_types

            # add negative entities and relations
            if self.config.max_neg_type_ratio > 0:
                neg_ratio = random.uniform(0.25, self.config.max_neg_type_ratio)

                # shuffle negative entities and relations
                random.shuffle(negative_entities)
                random.shuffle(negative_relations)

                # add negative entities and relations
                ent_types.extend(negative_entities[:int(len(negative_entities) * neg_ratio)])
                rel_types.extend(negative_relations[:int(len(negative_relations) * neg_ratio)])

            if self.config.shuffle_types:
                random.shuffle(ent_types)
                random.shuffle(rel_types)

            # random length of entities and relations
            if self.config.random_drop:
                sample_n_ent = random.randint(1, len(ent_types))
                sample_n_rel = random.randint(1, len(rel_types))

                ent_types = ent_types[:sample_n_ent]
                rel_types = rel_types[:sample_n_rel]

            ent_types = ent_types[:self.config.max_ent_types]
            rel_types = rel_types[:self.config.max_rel_types]

            ent_to_id, id_to_ent = self.create_mapping(ent_types)
            rel_to_id, id_to_rel = self.create_mapping(rel_types)

            all_ent_to_id.append(ent_to_id)
            all_id_to_ent.append(id_to_ent)
            all_rel_to_id.append(rel_to_id)
            all_id_to_rel.append(id_to_rel)

            negative_entities.extend(ent_types)
            negative_relations.extend(rel_types)

        return all_ent_to_id, all_id_to_ent, all_rel_to_id, all_id_to_rel

    def collate_fn(self, batch_list: List[Dict], entity_types: List[str] = None,
                   relation_types: List[str] = None) -> Dict:
        """Collate a batch of data."""

        if entity_types is None or relation_types is None:
            ent_to_id, id_to_ent, rel_to_id, id_to_rel = self.batch_generate_class_mappings(batch_list)
        else:
            ent_to_id, id_to_ent = self.create_mapping(entity_types)
            rel_to_id, id_to_rel = self.create_mapping(relation_types)

        batch = [self.preprocess_spans(b["tokenized_text"], b["entities"], b["relations"],
                                       ent_to_id if not isinstance(ent_to_id, list) else ent_to_id[i]) for i, b in
                 enumerate(batch_list)]

        return self.create_batch_dict(batch, ent_to_id, id_to_ent, rel_to_id, id_to_rel)

    def create_batch_dict(self, batch: List[Dict], ent_to_id: List[Dict[str, int]], id_to_ent: List[Dict[int, str]],
                          rel_to_id: List[Dict[str, int]], id_to_rel: List[Dict[int, str]]) -> Dict:
        """Create a dictionary for a batch of data."""

        # Extract necessary information from the batch
        tokens = [el["tokens"] for el in batch]
        span_idx = pad_sequence([b["span_idx"] for b in batch], batch_first=True, padding_value=0)
        span_label = pad_sequence([el["span_label"] for el in batch], batch_first=True, padding_value=-1)
        seq_length = torch.LongTensor([el["seq_length"] for el in batch])
        entities = [el["entities"] for el in batch]
        relations = [el["relations"] for el in batch]

        # Create a mask for valid spans
        span_mask = span_label != -1

        # Return a dictionary with the preprocessed spans
        return {
            'seq_length': seq_length,
            'span_idx': span_idx,
            'tokens': tokens,
            'span_mask': span_mask,
            'span_label': span_label,
            'entities': entities,
            'relations': relations,
            'ent_to_id': ent_to_id,
            'id_to_ent': id_to_ent,
            'rel_to_id': rel_to_id,
            'id_to_rel': id_to_rel
        }

    def create_dataloader(self, data, entity_types=None, relation_types=None, **kwargs) -> DataLoader:
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types, relation_types), **kwargs)


class TokenPromptProcessorTR:
    def __init__(self, entity_token, relation_token, sep_token):
        self.entity_token = entity_token
        self.sep_token = sep_token
        self.relation_token = relation_token

    def process(self, x, token_rep_layer, mode):
        if mode == "train":
            return self._process_train(x, token_rep_layer)
        elif mode == "eval":
            return self._process_eval(x, token_rep_layer)
        else:
            raise ValueError("Invalid mode specified. Choose 'train' or 'eval'.")

    def _process_train(self, x, token_rep_layer):

        device = next(token_rep_layer.parameters()).device

        new_length = x["seq_length"].clone()
        new_tokens = []
        all_len_prompt = []
        num_classes_all = []
        num_relations_all = []

        for i in range(len(x["tokens"])):
            all_types_i = list(x["ent_to_id"][i].keys())
            all_relations_i = list(x["rel_to_id"][i].keys())
            entity_prompt = []
            relation_prompt = []
            num_classes_all.append(len(all_types_i))
            num_relations_all.append(len(all_relations_i))

            for entity_type in all_types_i:
                entity_prompt.append(self.entity_token)
                entity_prompt.append(entity_type)
            entity_prompt.append(self.sep_token)

            for relation_type in all_relations_i:
                relation_prompt.append(self.relation_token)
                relation_prompt.append(relation_type)
            relation_prompt.append(self.sep_token)

            combined_prompt = entity_prompt + relation_prompt
            tokens_p = combined_prompt + x["tokens"][i]
            new_length[i] += len(combined_prompt)
            new_tokens.append(tokens_p)
            all_len_prompt.append(len(combined_prompt))

        max_num_classes = max(num_classes_all)
        entity_type_pos = torch.arange(max_num_classes).unsqueeze(0).expand(len(num_classes_all), -1).to(device)
        entity_type_mask = entity_type_pos < torch.tensor(num_classes_all).unsqueeze(-1).to(device)

        max_num_relations = max(num_relations_all)
        relation_type_pos = torch.arange(max_num_relations).unsqueeze(0).expand(len(num_relations_all), -1).to(device)
        relation_type_mask = relation_type_pos < torch.tensor(num_relations_all).unsqueeze(-1).to(device)

        bert_output = token_rep_layer(new_tokens, new_length)
        word_rep_w_prompt = bert_output["embeddings"]
        mask_w_prompt = bert_output["mask"]

        word_rep = []
        mask = []
        entity_type_rep = []
        relation_type_rep = []

        for i in range(len(x["tokens"])):
            prompt_entity_length = all_len_prompt[i]
            entity_len = 2 * len(list(x["ent_to_id"][i].keys())) + 1
            relation_len = 2 * len(list(x["rel_to_id"][i].keys())) + 1

            word_rep.append(word_rep_w_prompt[i, prompt_entity_length:new_length[i]])
            mask.append(mask_w_prompt[i, prompt_entity_length:new_length[i]])

            entity_rep = word_rep_w_prompt[i, :entity_len - 1]
            entity_rep = entity_rep[0::2]
            entity_type_rep.append(entity_rep)

            relation_rep = word_rep_w_prompt[i, entity_len:entity_len + relation_len - 1]
            relation_rep = relation_rep[0::2]
            relation_type_rep.append(relation_rep)

        word_rep = pad_sequence(word_rep, batch_first=True)
        mask = pad_sequence(mask, batch_first=True)
        entity_type_rep = pad_sequence(entity_type_rep, batch_first=True)
        relation_type_rep = pad_sequence(relation_type_rep, batch_first=True)

        return word_rep, mask, entity_type_rep, entity_type_mask, relation_type_rep, relation_type_mask

    def _process_eval(self, x, token_rep_layer):
        all_types = list(x["ent_to_id"].keys())
        all_relations = list(x["rel_to_id"].keys())
        entity_prompt = []
        relation_prompt = []

        for entity_type in all_types:
            entity_prompt.append(self.entity_token)
            entity_prompt.append(entity_type)
        entity_prompt.append(self.sep_token)

        for relation_type in all_relations:
            relation_prompt.append(self.relation_token)
            relation_prompt.append(relation_type)
        relation_prompt.append(self.sep_token)

        combined_prompt = entity_prompt + relation_prompt
        prompt_entity_length = len(combined_prompt)
        tokens_p = [combined_prompt + tokens for tokens in x["tokens"]]
        seq_length_p = x["seq_length"] + prompt_entity_length

        # Converting tokens_p to a format suitable for token_rep_layer
        out = token_rep_layer(tokens_p, seq_length_p)

        word_rep_w_prompt = out["embeddings"]
        mask_w_prompt = out["mask"]

        word_rep = word_rep_w_prompt[:, prompt_entity_length:, :]
        mask = mask_w_prompt[:, prompt_entity_length:]

        entity_type_rep = word_rep_w_prompt[:, :len(entity_prompt) - 1, :]
        entity_type_rep = entity_type_rep[:, 0::2, :]

        relation_type_rep = word_rep_w_prompt[:, len(entity_prompt):prompt_entity_length - 1, :]
        relation_type_rep = relation_type_rep[:, 0::2, :]

        return word_rep, mask, entity_type_rep, relation_type_rep
