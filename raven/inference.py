"""
MIT License

Copyright (c) 2026 0xf0xy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from core.training.train import train_model
import yaml

def main():
    with open("raven/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    model, vectorizers = train_model(config)

    def greedy_search(model, input_text, sequence_length):
        tokenized_input = vectorizers.input_vectorizer([input_text])
        decoded_sentence = "<BOS>"
        for i in range(sequence_length):
            tokenized_target = vectorizers.output_vectorizer([decoded_sentence])[:, :-1]
            predictions = model.predict([tokenized_input, tokenized_target])
            predicted_token_index = predictions[0][i, :].argmax()
            predicted_token = vectorizers.output_vectorizer.get_vocabulary()[
                predicted_token_index
            ]
            if predicted_token == "<EOS>":
                break
            decoded_sentence += " " + predicted_token
        return decoded_sentence

    while True:
        try:
            print(f"IA: {greedy_search(model, input('> '), sequence_length=128)}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
