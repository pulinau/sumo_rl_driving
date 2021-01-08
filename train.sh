#!/bin/bash

rm -f output.log || true
echo "Process starting on $(date)" 1>&2
python3 main.py --version=1 > output.log
echo "Process ended on $(date)" 1>&2
