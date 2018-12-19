#! /bin/bash
#
# Score system output (BPE and truecased) against reference (detokenized).
# Pass -lc for lowercased BLEU.

if [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
    echo "Usage: $0 [-lc] reference system"
    exit 1
fi

if [[ $1 == -* ]]; then
    if [ "$1" == "-lc" ]; then
	opt=$1
	shift
    else
	echo "invalid flag: $1"
	exit 1
    fi
fi

THIS_DIR=$(dirname $0)
REF=$1
SYS=$2

MOSES_ROOT=${THIS_DIR}/../third-party/mosesdecoder
cat $SYS \
| sed 's/$/ /' | sed 's/@@ //g' \
| $MOSES_ROOT/scripts/recaser/detruecase.perl \
| $MOSES_ROOT/scripts/tokenizer/detokenizer.perl \
| sed 's|@/@|/|g' | sed 's/@-/-/g' | sed 's/-@/-/g' | sed 's|</s>||' \
| ${THIS_DIR}/../third-party/nematus/data/multi-bleu-detok.perl $opt $REF
