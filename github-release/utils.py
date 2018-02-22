#!python3
__author__ = "Changjian Li"

import inspect

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}
