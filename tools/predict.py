#! /usr/bin/env python
#
# Run prediction.

import argparse
import logging
import os
import socket

import sys
import time

from cnn.cnn_mt import CNMT
from cnn.config import ModelConfiguration
from cnn.data_utils import x_batch_generator, x_prepare_data
from cnn.logging_utils import init_logging
from cnn.model_dir_utils import find_latest_model
from cnn.vocab import Vocab


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('src')
    parser.add_argument('tgt')
    parser.add_argument('--beam-width', type=int, default=10,
                        help='(default: %(default)s)')
    parser.add_argument('--max-words', type=int, default=80,
                        help='(default: %(default)s)')
    parser.add_argument('--model-filename',
                        help='use specific model instead of latest iter')
    parser.add_argument('--log-level', default='INFO',
                        help='(default: %(default)s')
    parser.add_argument('--log-file',
                        help='(default: predict-{tgt}.log)')
    parser.add_argument('--batch-greedy', action='store_true',
                        help='greedy decode on batches of sentences at once')
    parser.add_argument('--batch-size', type=int, default=80,
                        help='batch size for --batch-greedy '
                             '(default: %(default)s)')
    args = parser.parse_args()

    model_file = os.path.join(args.model_dir, args.model_filename) \
        if args.model_filename else find_latest_model(args.model_dir)
    log_file = args.log_file
    if not log_file:
        log_file = '{}.predict.log'.format(args.tgt)
    init_logging(log_file, args.log_level)
    log.info('command line args: {}'.format(args))

    config_path = os.path.join(args.model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise ValueError('no config file found in model directory')
    config = ModelConfiguration(config_path)

    log.info('hostname: %s', socket.gethostname())
    x_vocab = Vocab(vocab_path=os.path.join(args.model_dir, 'x_vocab.txt'))
    y_vocab = Vocab(vocab_path=os.path.join(args.model_dir, 'y_vocab.txt'))
    cnmt = CNMT(x_vocab, y_vocab, config)
    cnmt.load_params(model_file)

    if os.path.exists(args.tgt):
        orig_path = args.tgt
        args.tgt = orig_path + '.' + time.strftime('%Y%m%d%H%M%S')
        log.warning('{} already exists, writing {} instead'.format(orig_path, args.tgt))

    beam_width = 1 if args.batch_greedy else args.beam_width
    log.info('building batch predictor, beam width {}'.format(beam_width))
    f_predict = cnmt.build_batch_predictor()
    batch_predict(cnmt, f_predict, args.src, args.tgt, args.batch_size,
                  args.max_words, beam_width)


def batch_predict(cnmt, f_predict, src, tgt, batch_size, max_words, beam_width):
    log.info('begin batch prediction')
    batches = x_batch_generator(src, cnmt.x_vocab, batch_size, sys.maxsize)
    with open(tgt, 'w', encoding='utf8') as f:
        for x_in in batches:
            x, x_mask = x_prepare_data(x_in)
            result = f_predict(x, x_mask, max_words, beam_width)
            for sent in result:
                print(cnmt.y_vocab.words_for_indexes(sent), file=f)
    log.info('end prediction')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        exit(1)
