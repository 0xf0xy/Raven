import keras
import json


class Vectorizers:
    def __init__(self, config_file):
        self.config = config_file
        self.input_vectorizer = keras.layers.TextVectorization(
            self.config["model"]["vocab_size"],
            output_sequence_length=self.config["model"]["seq_len"],
        )
        self.output_vectorizer = keras.layers.TextVectorization(
            self.config["model"]["vocab_size"],
            None,
            output_sequence_length=self.config["model"]["seq_len"] + 1,
        )

    def adapt(self, inputs, outputs):
        self.input_vectorizer.adapt(inputs)
        self.output_vectorizer.adapt(outputs)

    def save_vocab(self):
        with open(self.config["path"]["vocab_path"][0], "w", encoding="utf-8") as file:
            for item in self.input_vectorizer.get_vocabulary():
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(self.config["path"]["vocab_path"][1], "w", encoding="utf-8") as file:
            for item in self.output_vectorizer.get_vocabulary():
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
