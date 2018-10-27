#!/bin/sh

# cleanup
rm -rf result*
rm -rf output.log eval.log
pkill -f main.py

for i in `seq 0 15`
do
  python3 main.py --play=True --version=$i 1>>output.log 2>eval.log
  pkill -f main.py
done
