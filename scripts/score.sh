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

# There's a bug in multi-bleu-detok.perl where the -lc flag does not
# consider non-ascii.  The version of the script in the cnn tree has
# been patched.  The `perl -C` flag below is a redundant workaround.
# Detruecasing should happen before detokenizing.

MOSES_ROOT=${THIS_DIR}/../third-party/mosesdecoder
cat $SYS \
| ( [[ -z "$opt" ]] && $MOSES_ROOT/scripts/recaser/detruecase.perl || cat ) \
| ${THIS_DIR}/detok.sh \
| perl -C ${THIS_DIR}/../third-party/nematus/data/multi-bleu-detok.perl $opt $REF
