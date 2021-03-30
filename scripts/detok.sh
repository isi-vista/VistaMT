#! /bin/bash
#
# Detokenize predictions.

THIS_DIR=$(dirname $0)
MOSES_ROOT=${THIS_DIR}/../third-party/mosesdecoder

$THIS_DIR/debpe.sh \
| $MOSES_ROOT/scripts/tokenizer/detokenizer.perl \
| sed 's/ @\([^@]\+\)@ /\1/g'
