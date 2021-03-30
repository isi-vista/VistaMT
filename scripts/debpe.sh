#! /bin/bash
#
# Remove bpe and sentence end artifacts.

sed 's|</s>||' | sed 's/$/ /' | sed 's/@@ //g'
