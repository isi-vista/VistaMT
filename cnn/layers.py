from math import sqrt
import tensorflow as tf


def embedding(name, layer_in, voc_size, dim, scale=0.1):
    with tf.variable_scope(name):
        e = tf.get_variable('e', shape=(voc_size, dim),
                            initializer=tf.random_normal_initializer(stddev=scale))
        return tf.nn.embedding_lookup(e, layer_in)


def linear(name, layer_in, d_in, d_out, dropout=0):
    with tf.variable_scope(name):
        std_w = sqrt((1.0 - dropout) / d_in)
        w = tf.get_variable('w', shape=(d_in, d_out),
                            initializer=tf.random_normal_initializer(stddev=std_w))
        b = tf.get_variable('b', shape=(d_out,),
                            initializer=tf.zeros_initializer())
        return tf.tensordot(layer_in, w, axes=1) + b


def conv_glu(name, layer_in, width, dim, padding, dropout):
    with tf.variable_scope(name):
        std_w = sqrt((4.0 * (1.0 - dropout)) / (dim * width))
        w = tf.get_variable('w', shape=(width, dim, dim),
                            initializer=tf.random_normal_initializer(stddev=std_w))
        b = tf.get_variable('b', shape=(dim,),
                            initializer=tf.zeros_initializer())
        wg = tf.get_variable('wg', shape=(width, dim, dim),
                             initializer=tf.random_normal_initializer(stddev=std_w))
        bg = tf.get_variable('bg', shape=(dim,),
                             initializer=tf.zeros_initializer())
        # convolve input
        block_out = tf.nn.conv1d(layer_in, w, stride=1, padding=padding) + b
        # compute gate activations
        gate_out = tf.nn.conv1d(layer_in, wg, stride=1, padding=padding) + bg
        # apply gates to outputs
        return block_out * tf.sigmoid(gate_out)
