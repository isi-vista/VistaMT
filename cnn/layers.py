from math import sqrt
import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    def __init__(self, voc_size, dim, scale=0.1):
        super(Embedding, self).__init__()
        initializer = tf.keras.initializers.TruncatedNormal(stddev=scale)
        self.e = tf.Variable(initializer(shape=(voc_size, dim)), trainable=True)

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.e, inputs)


class Linear(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, dropout_rate=0.):
        super(Linear, self).__init__()
        std_w = sqrt((1.0 - dropout_rate) / d_in)
        v_init = tf.keras.initializers.TruncatedNormal(stddev=std_w)
        self.v = tf.Variable(v_init(shape=(d_in, d_out)), trainable=True)
        self.g = tf.Variable(tf.norm(self.v, ord=2, axis=0), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(b_init(shape=(d_out,)), trainable=True)

    def call(self, inputs, **kwargs):
        v_norm = tf.norm(self.v, ord=2, axis=0)
        w = (self.g / v_norm) * self.v
        return tf.tensordot(inputs, w, axes=1) + self.b


class ConvGLU(tf.keras.layers.Layer):
    def __init__(self, width, dim, padding, dropout_rate):
        super(ConvGLU, self).__init__()
        self.width = width
        self.dim = dim
        self.padding = padding
        std_w = sqrt((4.0 * (1.0 - dropout_rate)) / (dim * width))
        v_init = tf.keras.initializers.TruncatedNormal(stddev=std_w)
        self.v = tf.Variable(v_init(shape=(width, dim, dim)), trainable=True)
        self.g = tf.Variable(tf.norm(tf.reshape(self.v, (dim * width, dim)), ord=2, axis=0),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(b_init(shape=(dim,)), trainable=True)
        self.vg = tf.Variable(v_init(shape=(width, dim, dim)), trainable=True)
        self.gg = tf.Variable(tf.norm(tf.reshape(self.v, (dim * width, dim)), ord=2, axis=0),
                              trainable=True)
        self.bg = tf.Variable(b_init(shape=(dim,)), trainable=True)

    def call(self, inputs, **kwargs):
        v_norm = tf.norm(tf.reshape(self.v, (self.dim * self.width, self.dim)), ord=2, axis=0)
        w = (self.g / v_norm) * self.v
        vg_norm = tf.norm(tf.reshape(self.vg, (self.dim * self.width, self.dim)), ord=2, axis=0)
        wg = (self.gg / vg_norm) * self.vg
        block_out = tf.nn.conv1d(inputs, w, stride=1, padding=self.padding) + self.b
        gate_out = tf.nn.conv1d(inputs, wg, stride=1, padding=self.padding) + self.bg
        return block_out * tf.sigmoid(gate_out)
