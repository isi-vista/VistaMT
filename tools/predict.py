#! /usr/bin/env python
#
# Run prediction.

import argparse
import logging
import os
import socket

from cnn import compat
from cnn.cnn_mt import ConvolutionalMT
from cnn.config import ModelConfiguration
from cnn.data_utils import XDataset
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
    parser.add_argument('--beam-size', type=int, default=10,
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
    parser.add_argument('--batch-max-words', type=int, default=4000,
                        help='(default: %(default)s)')
    parser.add_argument('--nbest', action='store_true',
                        help='write nbest list; n = beam-size')
    args = parser.parse_args()

    model_file = os.path.join(args.model_dir, args.model_filename) \
        if args.model_filename else find_latest_model(args.model_dir)
    log_file = args.log_file
    if not log_file:
        log_file = '{}.predict.log'.format(args.tgt)
    init_logging(log_file, args.log_level)
    log.info('command line args: {}'.format(args))

    if os.path.exists(args.tgt):
        raise ValueError('refusing to overwrite {}'.format(args.tgt))

    config_path = os.path.join(args.model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise ValueError('no config file found in model directory')
    config = ModelConfiguration(config_path)

    # need space for all words plus start and end tokens
    if args.max_words > config.num_positions - 2:
        raise ValueError('args.max-words must be <= config.num_positions - 2')

    log.info('hostname: %s', socket.gethostname())
    x_vocab = Vocab(vocab_path=os.path.join(args.model_dir, 'x_vocab.txt'))
    y_vocab = Vocab(vocab_path=os.path.join(args.model_dir, 'y_vocab.txt'))
    cnn_mt = ConvolutionalMT(config, x_vocab, y_vocab)
    compat.load_params(cnn_mt, model_file, expect_partial=True)

    beam_size = 1 if args.batch_greedy else args.beam_size
    dataset = XDataset(args.src, x_vocab, config.num_positions,
                       max_words_per_batch=args.batch_max_words,
                       max_sentences_per_batch=args.batch_size)
    log.info('building batch predictor, beam size {}'.format(beam_size))
    predict_f = batch_predict if not args.nbest else batch_predict_n
    predict_f(cnn_mt, dataset, args.tgt, args.max_words, beam_size)


def batch_predict(cnn_mt, dataset, tgt, max_words, beam_size):
    log.info('begin batch prediction')
    with open(tgt, 'w', encoding='utf8') as f:
        for x, x_mask in dataset():
            result = cnn_mt.predict(x, x_mask, max_words, beam_size)
            for sent in result:
                print(cnn_mt.y_vocab.words_for_indexes(sent), file=f)
    log.info('end prediction')


def batch_predict_n(cnmt, dataset, tgt, max_words, beam_size):
    log.info('begin batch nbest prediction')
    sent_idx = 0
    with open(tgt, 'w', encoding='utf8') as f:
        for x, x_mask in dataset():
            result = cnmt.predict_n(x, x_mask, max_words, beam_size)
            for batch_sent_idx, beam_sents in enumerate(result):
                for beam_sent_idx, (sent, cost) in enumerate(beam_sents):
                    words = cnmt.y_vocab.words_for_indexes(sent)
                    print(sent_idx, beam_sent_idx, cost, words, sep='\t', file=f)
                sent_idx += 1
    log.info('end prediction')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        exit(1)
