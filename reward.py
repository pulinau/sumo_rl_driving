#!/bin/python3

import numpy as np

def reward():
  r_safety = 0
  r_regulation = 0
  r_mission = 0
  r_comfort = 0
  return [r_safety, r_regulation, r_mission, r_comfort]
  
