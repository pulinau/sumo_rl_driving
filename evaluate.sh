#!/bin/sh

# cleanup
rm -rf result*
rm -rf output.log eval.log
pkill -f main.py

for i in `seq 0 15`
do
  echo "Running version $i"
  python3 main.py --play=False --version=$i 1>>output.log 2>eval.log
  cat eval.log
  pkill -f main.py
done
