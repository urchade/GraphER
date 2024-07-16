import numpy as np
import torch


def decode_relations(id_to_rel, logits, pair_indices, max_pairs, span_indices, threshold=0.5, output_confidence=False):
    """
    Decodes the relation logits into a list of relations.

    Args:
        id_to_rel (dict): Mapping from ID to relation.
        logits (Tensor): The relation logits.
        pair_indices (Tensor): The indices of the top-k pairs.
        max_pairs (int): The maximum number of pairs.
        span_indices (Tensor): The indices of the candidate spans.
        threshold (float, optional): The threshold for the sigmoid function. Defaults to 0.5.
        output_confidence (bool, optional): Whether to output the confidence. Defaults to False.

    Returns:
        list: A list of relations.
    """
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

def decode_entities(id_to_entity, logits, span_indices, threshold=0.5, output_confidence=False):
    """
    Decodes the entity logits into a list of entities.

    Args:
        id_to_entity (dict): Mapping from ID to entity.
        logits (Tensor): The entity logits.
        span_indices (Tensor): The indices of the candidate spans.
        threshold (float, optional): The threshold for the sigmoid function. Defaults to 0.5.
        output_confidence (bool, optional): Whether to output the confidence. Defaults to False.

    Returns:
        list: A list of entities.
    """
    # Apply sigmoid function to logits
    probabilities = torch.sigmoid(logits)

    # Initialize list of entities
    entities = []

    # Get indices where probability is greater than threshold
    above_threshold_indices = (probabilities > threshold).nonzero(as_tuple=True)

    # Iterate over indices where probability is greater than threshold
    for batch_idx, position, class_idx in zip(*above_threshold_indices):
        # Get entity label
        label = id_to_entity[class_idx.item() + 1]

        # Get confidence
        confidence = probabilities[batch_idx, position, class_idx].item()

        # Append entity to list
        if output_confidence:
            entities.append((tuple(span_indices[batch_idx, position].tolist()), label, confidence))
        else:
            entities.append((tuple(span_indices[batch_idx, position].tolist()), label))

    return entities

def er_decoder(x, entity_logits, rel_logits, topk_pair_idx, max_top_k, candidate_spans_idx, threshold=0.5,
            output_confidence=False):
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