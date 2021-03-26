import tensorflow as tf
import numpy as np


def positional_encoding(max_pos, dim, scale = .1):
    angle_rads = _get_angles(np.arange(max_pos)[:, np.newaxis],
                             np.arange(dim)[np.newaxis, :],
                             dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) * scale
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) * scale
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def _get_angles(pos, i, dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim))
    return pos * angle_rates
