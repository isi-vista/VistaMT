import numpy as np
import tensorflow as tf

from cnn.layers import Linear


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, emb_dim, dec_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        assert emb_dim % self.num_heads == 0
        self.depth = emb_dim // self.num_heads
        self.prj_in = Linear(dec_dim, emb_dim)
        self.prj_out = Linear(emb_dim, dec_dim)
        self.wq = tf.keras.layers.Dense(emb_dim)
        self.wk = tf.keras.layers.Dense(emb_dim)
        self.wv = tf.keras.layers.Dense(emb_dim)
        self.dense = tf.keras.layers.Dense(emb_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        ctx, ctx_plus_emb, x_mask, prev_w_emb, state_pre_attn = inputs
        x_len = tf.reduce_sum(x_mask, axis=1)
        q = self.prj_in(state_pre_attn)
        q = (q + prev_w_emb) / tf.sqrt(2.)
        k = ctx
        v = ctx_plus_emb
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, x_mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.emb_dim))

        output = self.dense(concat_attention)
        output_scale_factor = x_len / tf.sqrt(x_len)
        output_scale_factor = tf.expand_dims(tf.expand_dims(output_scale_factor, axis=1), axis=2)
        output = output * output_scale_factor
        output = self.prj_out(output)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        inf_mask = tf.where(tf.equal(mask, 1.), tf.zeros_like(mask, dtype=tf.float32),
                            tf.ones_like(mask, dtype=tf.float32) * np.NINF)
        scaled_attention_logits += tf.expand_dims(tf.expand_dims(inf_mask, axis=1), axis=2)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

