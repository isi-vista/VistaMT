import glob
import logging
import os
import tarfile
import numpy as np

from cnn.cnmt import CNMT_1

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_RANDOM_SEED = 1234
np.random.seed(_RANDOM_SEED)


class CNMT(object):
    def __init__(self, x_vocab, y_vocab, config):
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.cnmt = CNMT_1(config, x_vocab, y_vocab)
        log.info('ModelConfiguration: %s', config.to_json())

    def load_params(self, filename):
        log.info('Loading params from {}'.format(filename))
        tar_filename = filename
        tf_name = None
        with tarfile.open(tar_filename, 'r') as tar:
            for name in tar.getnames():
                i = name.find('.npz')
                if i > 0:
                    tf_name = os.path.join(os.path.dirname(filename), name[:i + 4])
                    log.info('Found tf model name inside tar: {}'.format(tf_name))
                    break
            tar.extractall(os.path.dirname(tar_filename))
        self.cnmt.load_params(tf_name)
        log.info('END Loading params')

    def save_params(self, filename):
        log.info('Saving params to {}'.format(filename))
        self.cnmt.save_params(filename)
        tar_filename = filename
        with tarfile.open(tar_filename, 'w') as tar:
            paths = glob.glob(filename + '.*')
            for path in paths:
                tar.add(path, arcname=os.path.basename(path))
        for path in paths:
            os.remove(path)
        log.info('END Saving params')

    def build_trainer(self, optimizer):
        def train(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_, lr_):
            self.cnmt.set_learning_rate(lr_)
            return self.cnmt.train(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_)

        def compute_cost(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_):
            return self.cnmt.get_cost(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_)

        def predict(x_, max_words_):
            x = [x_]
            x_mask = np.ones_like(x)
            return self.cnmt.predict(x, x_mask, max_words_, 1)[0], 0

        return train, compute_cost, predict

    def build_predictor(self):
        def predict(x_, beam_width, max_words):
            x = [x_]
            x_mask = np.ones_like(x)
            return self.cnmt.predict(x, x_mask, max_words, beam_width)[0], 0

        return predict

    def build_batch_predictor(self):
        def predict(x, x_mask, max_words, beam_width=1):
            return self.cnmt.predict(x, x_mask, max_words, beam_width)

        return predict
