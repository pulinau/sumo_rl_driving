#!/bin/sh

rm -rf result
rm -rf output.log eval.log

for i in `seq 0 6`
do
  python3 main.py --play=True --version=$i 1>>output.log 2>eval.log
done
