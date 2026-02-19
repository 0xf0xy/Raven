import json


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_raw_lists(path):
    inputs = []
    targets = []

    for sample in load_jsonl(path):
        inputs.append(sample["input"])
        targets.append("<BOS> " + sample["target"] + " <EOS>")

    return inputs, targets


def preprocess_data(path, vectorizers):
    inputs, targets = build_raw_lists(path)
    vectorizers.adapt(inputs, targets)
    encoder_inputs = vectorizers.input_vectorizer(inputs)
    target_tokens = vectorizers.output_vectorizer(targets)

    decoder_inputs = target_tokens[:, :-1]
    decoder_targets = target_tokens[:, 1:]

    return (
        encoder_inputs,
        decoder_inputs,
        decoder_targets,
    )
