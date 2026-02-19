from core.model.layers import PaddingMask, LookAheadMask, Encoder, Decoder
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class Transformer(keras.Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Transformer, self).__init__(**kwargs)
        self.seq_len = config["seq_len"]
        self.vocab_size = config["vocab_size"]
        self.num_layers = config["num_layers"]
        self.d_ffn = config["d_ffn"]
        self.d_model = config["d_model"]
        self.num_heads = config["num_heads"]
        self.dropout = config["dropout"]
        self.padding_mask = PaddingMask()
        self.look_ahead_mask = LookAheadMask()
        self.encoder = Encoder(
            self.seq_len,
            self.vocab_size,
            self.num_layers,
            self.d_ffn,
            self.d_model,
            self.num_heads,
            self.dropout,
        )
        self.decoder = Decoder(
            self.seq_len,
            self.vocab_size,
            self.num_layers,
            self.d_ffn,
            self.d_model,
            self.num_heads,
            self.dropout,
        )
        self.dense_decoder_output = keras.layers.Dense(self.vocab_size)

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "vocab_size": self.vocab_size,
                "num_layers": self.num_layers,
                "d_ffn": self.d_ffn,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )

        return config

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_mask = self.padding_mask(encoder_inputs)
        decoder_mask = tf.maximum(
            self.look_ahead_mask(decoder_inputs), self.padding_mask(decoder_inputs)
        )
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)
        decoder_outputs = self.decoder(
            decoder_inputs, encoder_outputs, decoder_mask, encoder_mask
        )

        return self.dense_decoder_output(decoder_outputs)
