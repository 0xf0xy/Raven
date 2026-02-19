import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class PaddingMask(keras.layers.Layer):
    def call(self, inputs):
        mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)

        return mask[:, tf.newaxis, tf.newaxis, :]


@keras.saving.register_keras_serializable()
class LookAheadMask(keras.layers.Layer):
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        return mask[tf.newaxis, tf.newaxis, :, :]


@keras.saving.register_keras_serializable()
class MultiheadAttention(keras.layers.Layer):
    def __init__(self, d_model, n_heads, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model

        assert d_model % self.n_heads == 0

        self.d_heads = d_model // self.n_heads
        self.scale = tf.math.sqrt(tf.cast(self.d_heads, tf.float32))
        self.query_dense = keras.layers.Dense(d_model)
        self.key_dense = keras.layers.Dense(d_model)
        self.value_dense = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.n_heads, self.d_heads))

        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask):
        score = tf.matmul(query, key, transpose_b=True)
        scaled_score = score / self.scale

        if mask is not None:
            scaled_score += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_score, axis=-1)

        return tf.matmul(attention_weights, value)

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        return self.dense(concat_attention)


@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, d_model, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(vocab_size, d_model)
        self.position_embeddings = keras.layers.Embedding(sequence_length, d_model)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        embedded_tokens *= tf.math.sqrt(tf.cast(self.token_embeddings.output_dim, tf.float32))

        return embedded_tokens + embedded_positions


@keras.saving.register_keras_serializable()
class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, ffn_dim, dropout, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.sequential = keras.Sequential(
            [
                keras.layers.Dense(ffn_dim, "relu"),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(d_model),
            ]
        )

    def call(self, inputs):
        return self.sequential(inputs)
    

@keras.saving.register_keras_serializable()
class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_ffn, d_model, n_heads, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        self.norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop_1 = keras.layers.Dropout(dropout)
        self.drop_2 = keras.layers.Dropout(dropout)

    def call(self, encoder_inputs, padding_mask):
        self_attention = self.self_attention(
            encoder_inputs, encoder_inputs, encoder_inputs, padding_mask
        )
        drop_1 = self.drop_1(self_attention)
        norm_1 = self.norm_1(encoder_inputs + drop_1)
        ffn = self.ffn(norm_1)
        drop_2 = self.drop_2(ffn)
        
        return self.norm_2(norm_1 + drop_2)


@keras.saving.register_keras_serializable()
class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_ffn, d_model, n_heads, dropout, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.cross_attention = MultiheadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        self.norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop_1 = keras.layers.Dropout(dropout)
        self.drop_2 = keras.layers.Dropout(dropout)
        self.drop_3 = keras.layers.Dropout(dropout)

    def call(self, decoder_inputs, encoder_outputs, look_ahead_mask, padding_mask):
        self_attention = self.self_attention(
            decoder_inputs, decoder_inputs, decoder_inputs, look_ahead_mask
        )
        drop_1 = self.drop_1(self_attention)
        norm_1 = self.norm_1(decoder_inputs + drop_1)
        cross_attention = self.cross_attention(
            norm_1, encoder_outputs, encoder_outputs, padding_mask
        )
        drop_2 = self.drop_2(cross_attention)
        norm_2 = self.norm_2(norm_1 + drop_2)
        ffn = self.ffn(norm_2)
        drop_3 = self.drop_3(ffn)
        
        return self.norm_3(norm_2 + drop_3)


@keras.saving.register_keras_serializable()
class Encoder(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        vocab_size,
        n_layers,
        d_ffn,
        d_model,
        n_heads,
        dropout,
        **kwargs,
    ):
        super(Encoder, self).__init__(**kwargs)
        self.token_and_position_embedding = TokenAndPositionEmbedding(
            sequence_length, vocab_size, d_model
        )
        self.dropout = keras.layers.Dropout(dropout)
        self.layers = [
            EncoderLayer(d_ffn, d_model, n_heads, dropout) for _ in range(n_layers)
        ]

    def call(self, encoder_inputs, padding_mask):
        embeddings = self.token_and_position_embedding(encoder_inputs)
        outputs = self.dropout(embeddings)

        for layer in self.layers:
            outputs = layer(outputs, padding_mask)

        return outputs


@keras.saving.register_keras_serializable()
class Decoder(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        vocab_size,
        n_layers,
        d_ffn,
        d_model,
        n_heads,
        dropout,
        **kwargs,
    ):
        super(Decoder, self).__init__(**kwargs)
        self.token_and_position_embedding = TokenAndPositionEmbedding(
            sequence_length, vocab_size, d_model
        )
        self.dropout = keras.layers.Dropout(dropout)
        self.layers = [
            DecoderLayer(d_ffn, d_model, n_heads, dropout) for _ in range(n_layers)
        ]

    def call(self, decoder_inputs, encoder_outputs, look_ahead_mask, padding_mask):
        embeddings = self.token_and_position_embedding(decoder_inputs)
        outputs = self.dropout(embeddings)

        for layer in self.layers:
            outputs = layer(outputs, encoder_outputs, look_ahead_mask, padding_mask)

        return outputs
