import numpy as np
import torch


def decode_relations(id_to_rel, logits, pair_indices, max_pairs, span_indices, threshold=0.5, output_confidence=False):
    # Apply sigmoid function to logits
    probabilities = torch.sigmoid(logits)

    # Initialize list of relations
    relations = [[] for _ in range(len(logits))]

    # Get indices where probability is greater than threshold
    above_threshold_indices = (probabilities > threshold).nonzero(as_tuple=True)

    # Iterate over indices where probability is greater than threshold
    for batch_idx, position, class_idx in zip(*above_threshold_indices):
        # Get relation label
        label = id_to_rel[class_idx.item() + 1]

        # Get predicted pair index
        predicted_pair_idx = pair_indices[batch_idx, position].item()

        # Unravel predicted pair index into head and tail
        head_idx, tail_idx = np.unravel_index(predicted_pair_idx, (max_pairs, max_pairs))

        # Convert head and tail indices to tuples
        head = tuple(span_indices[batch_idx, head_idx].tolist())
        tail = tuple(span_indices[batch_idx, tail_idx].tolist())

        # Get confidence
        confidence = probabilities[batch_idx, position, class_idx].item()

        # Append relation to list
        if output_confidence:
            relations[batch_idx.item()].append((head, tail, label, confidence))
        else:
            relations[batch_idx.item()].append((head, tail, label))

    return relations


def decode_entities(id_to_ent, logits, span_indices, threshold=0.5, output_confidence=False):
    # Apply sigmoid function to logits
    probabilities = torch.sigmoid(logits)

    # Initialize list of entities
    entities = []

    # Get indices where probability is greater than threshold
    above_threshold_indices = (probabilities > threshold).nonzero(as_tuple=True)

    # Iterate over indices where probability is greater than threshold
    for batch_idx, position, class_idx in zip(*above_threshold_indices):
        # Get entity label
        label = id_to_ent[class_idx.item() + 1]

        # Get confidence
        confidence = probabilities[batch_idx, position, class_idx].item()

        # Append entity to list
        if output_confidence:
            entities.append((tuple(span_indices[batch_idx, position].tolist()), label, confidence))
        else:
            entities.append((tuple(span_indices[batch_idx, position].tolist()), label))

    return entities


def er_decoder(x, entity_logits, rel_logits, topk_pair_idx, max_top_k, candidate_spans_idx, threshold=0.5,
               output_confidence=False, token_splitter=None):
    entities = decode_entities(x["id_to_ent"], entity_logits, candidate_spans_idx, threshold, output_confidence)
    relations = decode_relations(x["id_to_rel"], rel_logits, topk_pair_idx, max_top_k, candidate_spans_idx, threshold,
                                 output_confidence)
    return entities, relations


def get_relation_with_span(x):
    entities, relations = x['entities'], x['relations']
    B = len(entities)
    relation_with_span = [[] for i in range(B)]
    for i in range(B):
        rel_i = relations[i]
        ent_i = entities[i]
        for rel in rel_i:
            act = (ent_i[rel[0]], ent_i[rel[1]], rel[2])
            relation_with_span[i].append(act)
    return relation_with_span


def get_ground_truth_relations(x, candidate_spans_idx, candidate_span_label):
    B, max_top_k = candidate_span_label.shape

    relation_classes = torch.zeros((B, max_top_k, max_top_k), dtype=torch.long, device=candidate_spans_idx.device)

    # Populate relation classes
    for i in range(B):
        rel_i = x["relations"][i]
        ent_i = x["entities"][i]

        new_heads, new_tails, new_rel_type = [], [], []

        # Loop over the relations and entities to populate initial lists
        for k in rel_i:
            heads_i = [ent_i[k[0]][0], ent_i[k[0]][1]]
            tails_i = [ent_i[k[1]][0], ent_i[k[1]][1]]
            type_i = k[2]
            new_heads.append(heads_i)
            new_tails.append(tails_i)
            new_rel_type.append(type_i)

        # Update the original lists
        heads_, tails_, rel_type = new_heads, new_tails, new_rel_type

        # idx of candidate spans
        cand_i = candidate_spans_idx[i].tolist()

        for heads_i, tails_i, type_i in zip(heads_, tails_, rel_type):

            flag = False
            if isinstance(x["rel_to_id"], dict):
                if type_i in x["rel_to_id"]:
                    flag = True
            elif isinstance(x["rel_to_id"], list):
                if type_i in x["rel_to_id"][i]:
                    flag = True

            if heads_i in cand_i and tails_i in cand_i and flag:
                idx_head = cand_i.index(heads_i)
                idx_tail = cand_i.index(tails_i)

                if isinstance(x["rel_to_id"], list):
                    relation_classes[i, idx_head, idx_tail] = x["rel_to_id"][i][type_i]
                elif isinstance(x["rel_to_id"], dict):
                    relation_classes[i, idx_head, idx_tail] = x["rel_to_id"][type_i]

    # flat relation classes
    relation_classes = relation_classes.view(-1, max_top_k * max_top_k)

    # put to -1 class where corresponding candidate_span_label is -1 (for both head and tail)
    head_candidate_span_label = candidate_span_label.view(B, max_top_k, 1).repeat(1, 1, max_top_k).view(B, -1)
    tail_candidate_span_label = candidate_span_label.view(B, 1, max_top_k).repeat(1, max_top_k, 1).view(B, -1)

    relation_classes.masked_fill_(head_candidate_span_label.view(B, max_top_k * max_top_k) == -1, -1)  # head
    relation_classes.masked_fill_(tail_candidate_span_label.view(B, max_top_k * max_top_k) == -1, -1)  # tail

    return relation_classes


def get_candidates(sorted_idx, tensor_elem, topk=10):
    # sorted_idx [B, num_spans]
    # tensor_elem [B, num_spans, D] or [B, num_spans]

    sorted_topk_idx = sorted_idx[:, :topk]

    if len(tensor_elem.shape) == 3:
        B, num_spans, D = tensor_elem.shape
        topk_tensor_elem = tensor_elem.gather(1, sorted_topk_idx.unsqueeze(-1).expand(-1, -1, D))
    else:
        # [B, topk]
        topk_tensor_elem = tensor_elem.gather(1, sorted_topk_idx)

    return topk_tensor_elem, sorted_topk_idx
