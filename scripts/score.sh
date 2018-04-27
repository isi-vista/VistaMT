#! /bin/bash
#
# Score system output (BPE and truecased) against reference (detokenized).

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 reference system"
    exit 1
fi

THIS_DIR=$(dirname $0)
REF=$1
SYS=$2

MOSES_ROOT=${THIS_DIR}/../third-party/mosesdecoder
cat $SYS \
| sed 's/$/ /' | sed 's/\@\@ //g' | sed 's/ \@-\@ /-/' | sed 's|</s>||' \
| $MOSES_ROOT/scripts/recaser/detruecase.perl \
| $MOSES_ROOT/scripts/tokenizer/detokenizer.perl >$SYS.detok

cat $SYS.detok \
| ${THIS_DIR}/../third-party/nematus/data/multi-bleu-detok.perl $REF \
| tee $SYS.score
