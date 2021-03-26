import os
import tensorflow as tf
import numpy as np

from cnn.beam_search import BeamSearch
from cnn.decoder import Decoder
from cnn.encoder import Encoder


_RANDOM_SEED = 1234
np.random.seed(_RANDOM_SEED)

DEFAULT_LEARNING_RATE = 0.0002
MAX_GRAD_NORM = 0.25

# Tensorflow will try to use all cores if it doesn't have a GPU.  If
# you forget to reserve one, you might not realize why things are so
# slow.  Detect this early.  If you really mean to use CPU, set the
# environment variable CNN_USE_CPU=1.
if 'CNN_USE_CPU' not in os.environ:
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('No CUDA GPU available.  To force CPU, set CNN_USE_CPU in environment')


class ConvolutionalMT:
    def __init__(self, config, x_vocab, y_vocab):
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        emb_dim = config.emb_dim
        encoder_arch = config.encoder_arch
        decoder_arch = config.decoder_arch
        out_emb_dim = config.out_emb_dim
        dropout_rate = config.dropout_rate
        num_positions = config.num_positions
        num_attn_heads = config.num_attn_heads
        num_dec_layers = 0
        for a in config.decoder_arch:
            num_dec_layers += a[0]
        self.encoder = Encoder(x_vocab.size(), emb_dim, encoder_arch, dropout_rate, num_dec_layers,
                               num_positions)
        self.decoder = Decoder(y_vocab.size(), emb_dim, out_emb_dim, decoder_arch, dropout_rate,
                               num_positions, num_attn_heads)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)
        self.beam_search = BeamSearch(self.encoder, self.decoder, y_vocab)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder,
                                              decoder=self.decoder)

    def load_params(self, file_prefix, expect_partial=False):
        if expect_partial:
            self.checkpoint.restore(file_prefix).expect_partial()
        else:
            self.checkpoint.restore(file_prefix)

    def save_params(self, file_prefix):
        self.checkpoint.write(file_prefix)

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    @tf.function(input_signature=(tf.TensorSpec((None, None), dtype=tf.int32),
                                  tf.TensorSpec((None, None), dtype=tf.float32),
                                  tf.TensorSpec((None, None), dtype=tf.int32),
                                  tf.TensorSpec((None, None), dtype=tf.float32)))
    def train(self, x, x_mask, y, y_mask):
        with tf.GradientTape() as tape:
            ctx, ctx_plus_emb = self.encoder([x, x_mask], training=True)
            y_out, _ = self.decoder([y[:, :-1], ctx, ctx_plus_emb, x_mask, None], training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:, 1:], logits=y_out)
            loss = loss * y_mask
            loss = tf.reduce_sum(loss) / tf.reduce_sum(y_mask)
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            gradients = [tf.clip_by_norm(grad, MAX_GRAD_NORM) for grad in gradients]
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss

    def predict(self, x, x_mask, maxlen, beam_size):
        return self.beam_search.predict(x, x_mask, maxlen, beam_size)

    def predict_n(self, x, x_mask, maxlen, beam_size):
        return self.beam_search.predict_n(x, x_mask, maxlen, beam_size)

    @tf.function(input_signature=(tf.TensorSpec((None, None), dtype=tf.int32),
                                  tf.TensorSpec((None, None), dtype=tf.float32),
                                  tf.TensorSpec((None, None), dtype=tf.int32),
                                  tf.TensorSpec((None, None), dtype=tf.float32)))
    def get_cost(self, x, x_mask, y, y_mask):
        ctx, ctx_plus_emb = self.encoder([x, x_mask], training=False)
        y_out, _ = self.decoder([y[:, :-1], ctx, ctx_plus_emb, x_mask, None], training=False)
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:, 1:], logits=y_out)
        cost = cost * y_mask
        cost = tf.reduce_sum(cost) / tf.reduce_sum(y_mask)
        return cost
