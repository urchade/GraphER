from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


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
            span_indices.extend([(i, i + j) for j in range(self.max_width)])

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

        for b in batch_list:
            ent_types = list(set([el[-1] for el in b['entities']]))
            rel_types = list(set([el[-1] for el in b['relations']]))

            ent_to_id, id_to_ent = self.create_mapping(ent_types)
            rel_to_id, id_to_rel = self.create_mapping(rel_types)

            all_ent_to_id.append(ent_to_id)
            all_id_to_ent.append(id_to_ent)
            all_rel_to_id.append(rel_to_id)
            all_id_to_rel.append(id_to_rel)

        return all_ent_to_id, all_id_to_ent, all_rel_to_id, all_id_to_rel

    def collate_fn(self, batch_list: List[Dict], entity_types: List[str] = None,
                   relation_types: List[str] = None) -> Dict:
        """Collate a batch of data."""

        if entity_types is None or relation_types is None:
            ent_to_id, id_to_ent, rel_to_id, id_to_rel = self.batch_generate_class_mappings(batch_list)
        else:
            ent_to_id, id_to_ent = self.create_mapping(entity_types)
            rel_to_id, id_to_rel = self.create_mapping(relation_types)

        batch = [self.preprocess_spans(b["tokenized_text"], b["entities"], b["relations"], ent_to_id[i]) for i, b in
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
