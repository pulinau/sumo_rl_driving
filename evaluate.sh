#!/bin/sh

for i in {0..9}
do
  python3 main.py --play=True --version=$i 2>>eval.log
done