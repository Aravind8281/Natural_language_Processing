import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, Embedding, Dense

class SelfAttention(Layer):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.query = Dense(embed_size)
        self.key = Dense(embed_size)
        self.value = Dense(embed_size)
        self.combine_heads = Dense(embed_size)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query(inputs)  # (batch_size, seq_len, embed_size)
        key = self.key(inputs)
        value = self.value(inputs)

        query = tf.reshape(
            query, (batch_size, -1, self.heads, self.head_dim)
        )  # (batch_size, seq_len, heads, head_dim)
        key = tf.reshape(key, (batch_size, -1, self.heads, self.head_dim))
        value = tf.reshape(value, (batch_size, -1, self.heads, self.head_dim))

        query = tf.transpose(query, perm=[0, 2, 1, 3])  # (batch_size, heads, seq_len, head_dim)
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, heads, seq_len, seq_len)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, dtype=tf.float32))

        attention = tf.nn.softmax(scores, axis=3)

        out = tf.matmul(attention, value)  # (batch_size, heads, seq_len, head_dim)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.embed_size))

        out = self.combine_heads(out)
        return out

class TransformerBlock(Layer):
    def __init__(self, embed_size, heads, forward_expansion, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = Self
