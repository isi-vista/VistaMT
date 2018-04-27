import sys
import operator
from collections import defaultdict


class Vocab(object):
    UNK = '<unk>'
    SENT_START = '<s>'
    SENT_END ='</s>'
    RESERVED = {UNK, SENT_START, SENT_END}

    def __init__(self, sentences_paths=None, vocab_path=None):
        if sentences_paths and vocab_path:
            raise ValueError('cannot pass both sentences_paths and vocab_path')
        self._words = []
        self._word2index = {}
        if sentences_paths:
            self._from_sentences(sentences_paths)
        elif vocab_path:
            self._from_vocab_file(vocab_path)
        else:
            raise ValueError('must pass one of sentences_paths or vocab_path')
        self.unk_index = self._word2index.get(Vocab.UNK)
        
    def _from_sentences(self, sentences_paths):
        self._words.append(Vocab.UNK)
        self._words.append(Vocab.SENT_START)
        self._words.append(Vocab.SENT_END)
        counts = defaultdict(int)
        for path in sentences_paths:
            with open(path) as f:
                for line in f:
                    for word in line.rstrip().split():
                        if not word in Vocab.RESERVED:
                            counts[word] += 1
        sorted_items = sorted(counts.items(), key=operator.itemgetter(1),
                              reverse=True)
        self._words.extend([x[0] for x in sorted_items])
        for i, word in enumerate(self._words):
            self._word2index[word] = i

    def _from_vocab_file(self, vocab_path):
        with open(vocab_path) as f:
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
        with open(path, 'w') as f:
            for word in self._words:
                print(word, file=f)


if __name__ == '__main__':
    vocab = Vocab(sentences_paths=[sys.argv[1]])
    print(vocab.lookup(Vocab.SENT_START))
    print(vocab.lookup(Vocab.SENT_END))
    print(vocab.word_for_index(0))
    print(vocab.lookup('than'))
    print(vocab.word_for_index(6))
    print(vocab.lookup('adsfdfd'))
    print(vocab.word_for_index(vocab.unk_index))
    vocab.write('vocab.txt')

    vocab = Vocab(vocab_path='vocab.txt')
    print(vocab.lookup(Vocab.SENT_START))
    print(vocab.lookup(Vocab.SENT_END))
    print(vocab.word_for_index(0))
    print(vocab.lookup('than'))
    print(vocab.word_for_index(6))
