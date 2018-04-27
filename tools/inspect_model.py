#! /usr/bin/env python
#
# Dump the binary model contents to standard output.

import numpy as np
import sys

if len(sys.argv) != 2:
    print('Usage: {} model_path'.format(sys.argv[0]), file=sys.stderr)
    exit(1)

for k, v in np.load(sys.argv[1]).items():
    print(k, v)
