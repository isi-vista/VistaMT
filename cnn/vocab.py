import argparse
import sys
import operator
from collections import OrderedDict


class Vocab(object):
    SENT_END ='</s>'
    SENT_START = '<s>'
    UNK = '<unk>'
    RESERVED = {SENT_END, SENT_START, UNK}

    def __init__(self, sentences_paths=None, vocab_path=None, min_words=1):
        if sentences_paths and vocab_path:
            raise ValueError('cannot pass both sentences_paths and vocab_path')
        self._words = []
        self._word2index = {}
        if sentences_paths:
            self._from_sentences(sentences_paths, min_words)
        elif vocab_path:
            self._from_vocab_file(vocab_path)
        else:
            raise ValueError('must pass one of sentences_paths or vocab_path')
        self.unk_index = self._word2index.get(Vocab.UNK)

    def _from_sentences(self, sentences_paths, min_words):
        self._words.append(Vocab.SENT_END) # SENT_END must have value 0
        self._words.append(Vocab.SENT_START)
        self._words.append(Vocab.UNK)
        counts = OrderedDict()
        for path in sentences_paths:
            with open(path, 'r', encoding='utf8') as f:
                for line in f:
                    for word in line.rstrip().split():
                        if not word in Vocab.RESERVED:
                            if word not in counts:
                                counts[word] = 1
                            else:
                                counts[word] += 1
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        self._words.extend([x[0] for x in sorted_items if x[1] >= min_words])
        for i, word in enumerate(self._words):
            self._word2index[word] = i

    def _from_vocab_file(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                word = line.rstrip()
                self._words.append(word)
                self._word2index[word] = i

    def size(self):
        return len(self._words)

    def lookup(self, word):
        return self._word2index.get(word, self.unk_index)

    def word_for_index(self, index):
        return self._words[index]

    def words_for_indexes(self, indexes, joiner=' '):
        return joiner.join([self.word_for_index(x) for x in indexes])

    def write(self, path):
        with open(path, 'w', encoding='utf8') as f:
            for word in self._words:
                print(word, file=f)

    def __eq__(self, other):
        return self._words == other._words and self._word2index == other._word2index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sentences_path', help='paths to tokenized sentences')
    parser.add_argument('output_path', help='output path')
    parser.add_argument('--min-words', type=int, default=1)
    args = parser.parse_args()
    vocab = Vocab([args.sentences_path], min_words=args.min_words)
    vocab.write(args.output_path)
