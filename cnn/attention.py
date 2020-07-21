import numpy as np
import tensorflow as tf

from cnn.layers import Linear


class Attention(tf.keras.layers.Layer):
    def __init__(self, emb_dim, dec_dim):
        super(Attention, self).__init__()
        self.prj_in = Linear(dec_dim, emb_dim)
        self.prj_out = Linear(emb_dim, dec_dim)

    def call(self, inputs, **kwargs):
        ctx, ctx_plus_emb, x_mask, prev_w_emb, state_pre_attn = inputs
        x_len = tf.reduce_sum(x_mask, axis=1)
        state_attn = self.prj_in(state_pre_attn)
        state_attn = (state_attn + prev_w_emb) / tf.sqrt(2.)
        alpha = tf.einsum('bye,bxe->byx', state_attn, ctx)
        inf_mask = tf.where(tf.equal(x_mask, 1.), tf.zeros_like(x_mask, dtype=tf.float32),
                            tf.ones_like(x_mask, dtype=tf.float32) * np.NINF)
        alpha = alpha + tf.expand_dims(inf_mask, axis=1)
        alpha = alpha - tf.reduce_max(alpha, axis=2, keepdims=True)
        alpha = tf.exp(alpha)
        alpha = alpha * tf.expand_dims(x_mask, axis=1)
        alpha = alpha / tf.reduce_sum(alpha, axis=2, keepdims=True)
        ctx_contrib = tf.expand_dims(alpha, axis=3) * tf.expand_dims(ctx_plus_emb, axis=1)
        ctx_contrib = tf.reduce_sum(ctx_contrib, axis=2)
        ctx_scale_factor = x_len / tf.sqrt(x_len)
        ctx_scale_factor = tf.expand_dims(tf.expand_dims(ctx_scale_factor, axis=1), axis=2)
        ctx_attn = ctx_contrib * ctx_scale_factor
        ctx_attn = self.prj_out(ctx_attn)
        return ctx_attn
