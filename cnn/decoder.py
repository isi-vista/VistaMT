import tensorflow as tf

from cnn.multi_head_attention import MultiHeadAttention
from cnn.layers import ConvGLU, Linear, Embedding
from cnn.positional_encoding import positional_encoding


class Decoder(tf.keras.Model):
    def __init__(self, voc_size, emb_dim, out_emb_dim, architecture, dropout_rate, num_positions,
                 num_attn_heads):
        super(Decoder, self).__init__()
        self.architecture = architecture
        self.emb_dim = emb_dim
        self.y_emb = Embedding(voc_size, emb_dim)
        self.pos_encoding = positional_encoding(num_positions, emb_dim)
        self.dropout_in = tf.keras.layers.Dropout(dropout_rate)
        self.conv_in_projections = []
        self.convolutions = []
        curr_dim = architecture[0][2]
        self.prj_in = Linear(emb_dim, curr_dim, dropout_rate)
        for depth, width, dim in architecture:
            if dim != curr_dim:
                self.conv_in_projections.append(Linear(curr_dim, dim))
            else:
                self.conv_in_projections.append(None)
            self.convolutions.append(DecoderStack(depth, width, dim, emb_dim, num_attn_heads,
                                                  dropout_rate))
            curr_dim = dim
        self.prj_out_1 = Linear(curr_dim, out_emb_dim)
        self.dropout_out = tf.keras.layers.Dropout(dropout_rate)
        self.prj_out_2 = Linear(out_emb_dim, voc_size, dropout_rate)

    def call(self, inputs, step_mode=False, pos=0, training=False, **kwargs):
        y_in, ctx, ctx_plus_emb, x_mask, prev_state = inputs
        if step_mode:
            y_in = tf.expand_dims(y_in, axis=1)
        seq_len = tf.shape(y_in)[1]
        next_state = [] if step_mode else None
        y_emb = self.y_emb(y_in)
        y_emb += self.pos_encoding[:, pos:pos+seq_len, :]
        if step_mode:
            y_emb = tf.ensure_shape(y_emb, (None, 1, self.emb_dim))
        y_emb = self.dropout_in(y_emb, training)
        h_dec = self.prj_in(y_emb)
        for idx in range(len(self.convolutions)):
            prj = self.conv_in_projections[idx]
            if prj:
                h_dec = prj(h_dec)
            stack_prev_state = prev_state[idx] if step_mode else None
            conv = self.convolutions[idx]
            h_dec, stack_next_state = conv([h_dec, ctx, ctx_plus_emb, x_mask, y_emb,
                                            stack_prev_state],
                                           step_mode=step_mode, training=training)
            if step_mode:
                next_state.append(stack_next_state)
        next_state = tuple(next_state) if next_state is not None else None
        dec_out = self.prj_out_1(h_dec)
        dec_out = self.dropout_out(dec_out, training)
        logits = self.prj_out_2(dec_out)
        return logits, next_state


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, depth, width, dim, emb_dim, num_attn_heads, dropout_rate):
        super(DecoderStack, self).__init__()
        self.depth = depth
        self.layers = []
        for _ in range(depth):
            self.layers.append(DecoderBlock(width, dim, emb_dim, num_attn_heads, dropout_rate))

    def call(self, inputs, step_mode=False, training=False):
        stack_in, ctx, ctx_plus_emb, x_mask, y_emb, prev_state = inputs
        h_stack = stack_in
        next_state = [] if step_mode else None
        for i in range(self.depth):
            layer = self.layers[i]
            blk_prev_state = prev_state[i] if step_mode else None
            h_stack, blk_next_state = layer([h_stack, ctx, ctx_plus_emb, x_mask, y_emb,
                                             blk_prev_state],
                                            step_mode=step_mode, training=training)
            if step_mode:
                next_state.append(blk_next_state)
        next_state = tuple(next_state) if next_state is not None else None
        return h_stack, next_state


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, width, dim, emb_dim, num_attn_heads, dropout_rate):
        super(DecoderBlock, self).__init__()
        self.width = width
        self.dim = dim
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.conv_glu = ConvGLU(width, dim, 'VALID', dropout_rate)
        self.attention = MultiHeadAttention(emb_dim, dim, num_attn_heads)

    def call(self, inputs, step_mode=False, training=False):
        block_in, ctx, ctx_plus_emb, x_mask, y_emb, prev_state = inputs
        batch_size = tf.shape(block_in)[0]
        conv_in = self.dropout(block_in, training=training)
        if not step_mode:
            prev_state = tf.zeros((batch_size, self.width - 1, self.dim))
        conv_in = tf.concat((prev_state, conv_in), axis=1)
        next_state = None if not step_mode else conv_in[:, 1:self.width, :]
        if step_mode:
            next_state = tf.ensure_shape(next_state, (None, self.width - 1, self.dim))
        state_pre_attn = self.conv_glu(conv_in)
        ctx_attn = self.attention([ctx, ctx_plus_emb, x_mask, y_emb, state_pre_attn])
        state_combined = (state_pre_attn + ctx_attn) / tf.sqrt(2.)
        y = (state_combined + block_in) / tf.sqrt(2.)
        return y, next_state
