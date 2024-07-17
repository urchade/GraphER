import torch

from model import GraphER


def save_model(current_model, path):
    config = current_model.config
    dict_save = {"model_weights": current_model.state_dict(), "config": config}
    torch.save(dict_save, path)


def load_model(path, model_name=None):
    dict_load = torch.load(path, map_location=torch.device('cpu'))
    config = dict_load["config"]

    if model_name is not None:
        config.model_name = model_name

    loaded_model = GraphER(config)
    loaded_model.load_state_dict(dict_load["model_weights"])
    return loaded_model


if __name__ == "__main__":
    import re

    model = load_model("/Users/urchadezaratiana/Documents/remote-server/ZerEL/fresh/deberta-v3-large_0/model_15000",
                       model_name="microsoft/deberta-v3-large")


    def process_text_and_relations(text, relation_types, entity_types=None, constraints=None):
        # Tokenize the text
        def tokenize_text(text):
            return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

        # Remove line breaks from the text and tokenize
        tokens = tokenize_text(text.replace("\n", ""))

        # Prepare the input for the model
        input_x = {"tokenized_text": tokens, "spans": [], "relations": [], "entities": []}

        # Call the model's collate function
        x = model.data_proc.collate_fn([input_x], relation_types=relation_types, entity_types=entity_types)

        # print(x)

        # Predict using the model
        entities, out = model.predict(x, threshold=0.1, output_confidence=True)

        ent_txt = {}

        for entity in entities:
            (start, end), etype, conf = entity

            print(" ".join(tokens[start:end + 1]), "--", etype, "-- confidence:", conf)

            ent_txt[" ".join(tokens[start:end + 1])] = etype

        # Process the output to extract and print relations and their confidence
        res = []
        for el in out[0]:
            (s_h, e_h), (s_t, e_t), rtype, conf = el
            head_ = " ".join(tokens[s_h:e_h + 1])
            tail_ = " ".join(tokens[s_t:e_t + 1])

            try:
                head_type = ent_txt[head_]
                tail_type = ent_txt[tail_]
            except:
                continue

            if constraints is not None:
                if (head_type, rtype, tail_type) not in constraints:
                    continue

            if (head_, rtype, tail_) in res:
                continue
            res.append((head_, rtype, tail_))
            print(head_, "---", rtype, "--", tail_, "-- confidence:", conf)


    # Example usage
    text = """
    Named Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast, Large Language Models (LLMs) can extract arbitrary entities through natural language instructions, offering greater flexibility. However, their size and cost, particularly for those accessed via APIs like ChatGPT, make them impractical in resource-limited scenarios. In this paper, we introduce a compact NER model trained to identify any type of entity. Leveraging a bidirectional transformer encoder, our model, GLiNER, facilitates parallel entity extraction, an advantage over the slow sequential token generation of LLMs. Through comprehensive testing, GLiNER demonstrate strong performance, outperforming both ChatGPT and fine-tuned LLMs in zero-shot evaluations on various NER benchmarks.
    """

    entity_types = ["model name", "task", "dataset", "metric"]
    relation_types = ["evaluated on", "used for", "outperforms", "hypernym of"]

    # constraints triplets
    constraints = [
        ("model name", "evaluated on", "dataset"),
        ("model name", "used for", "task"),
        ("model name", "outperforms", "model name"),
    ]

    # Parallel prediction
    print("Parallel prediction:")
    process_text_and_relations(text, relation_types, entity_types)

    # One-by-one prediction
    # print("\nOne by one prediction (should work better):")
    # for rel in relation_types:
    #    process_text_and_relations(text, [rel])
