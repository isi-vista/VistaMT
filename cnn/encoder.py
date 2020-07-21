import tensorflow as tf

from cnn.layers import ConvGLU, Linear, Embedding


class Encoder(tf.keras.Model):
    def __init__(self, voc_size, emb_dim, architecture, dropout_rate, num_dec_layers):
        super(Encoder, self).__init__()
        self.architecture = architecture
        self.num_decoder_layers = num_dec_layers
        self.x_emb = Embedding(voc_size, emb_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.conv_in_projections = []
        self.convolutions = []
        curr_dim = architecture[0][2]
        self.prj_in = Linear(emb_dim, curr_dim, dropout_rate)
        for depth, width, dim in architecture:
            if dim != curr_dim:
                self.conv_in_projections.append(Linear(curr_dim, dim))
            else:
                self.conv_in_projections.append(None)
            self.convolutions.append(EncoderStack(depth, width, dim, dropout_rate))
            curr_dim = dim
        self.prj_out = Linear(curr_dim, emb_dim)

    def call(self, inputs, training=False, **kwargs):
        x, x_mask = inputs
        x_emb = self.x_emb(x)
        x_emb = x_emb * tf.expand_dims(x_mask, axis=2)
        x_emb = self.dropout(x_emb, training)
        h_enc = self.prj_in(x_emb)
        for prj, conv in zip(self.conv_in_projections, self.convolutions):
            if prj:
                h_enc = prj(h_enc)
            h_enc = conv([h_enc, x_mask], training=training)
        ctx = self.prj_out(h_enc)
        scale_factor = 1. / (2 * self.num_decoder_layers)
        ctx = (scale_factor * ctx) + ((1 - scale_factor) * tf.stop_gradient(ctx))
        ctx_plus_emb = (ctx + x_emb) / tf.sqrt(2.)
        return ctx, ctx_plus_emb


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, depth, width, dim, dropout_rate):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(EncoderBlock(width, dim, dropout_rate))

    def call(self, inputs, training=False):
        stack_in, x_mask = inputs
        h_stack = stack_in
        for layer in self.layers:
            h_stack = layer([h_stack, x_mask], training=training)
        return h_stack


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, width, dim, dropout_rate):
        super(EncoderBlock, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.conv_glu = ConvGLU(width, dim, 'SAME', dropout_rate)

    def call(self, inputs, training=False):
        block_in, x_mask = inputs
        conv_in = self.dropout(block_in, training=training)
        conv_in = conv_in * tf.expand_dims(x_mask, axis=2)
        conv_out = self.conv_glu(conv_in)
        return (conv_out + block_in) / tf.sqrt(2.)
