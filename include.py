import sys
SUMO_TOOLS_DIR = "/usr/local/share/sumo/tools"
try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import traci
import sumolib

import numpy as np

import gym
from gym import spaces

from enum import Enum, auto
# simulation
class EnvState(Enum):
  DONE = auto()
  CRASH = auto()
  NORMAL = auto()
  NOT_STARTED = auto()
  ERROR = auto()

# action
class ActionLaneChange(Enum):
  NOOP = 0
  LEFT = 1
  RIGHT = 2

class ActionAccel(Enum):
  MAXDECEL = 0
  MEDDECEL = 1
  MINDECEL = 2
  NOOP = 3
  MINACCEL = 4
  MEDACCEL = 5
  MAXACCEL = 6

action_size = len(ActionLaneChange) * len(ActionAccel)
reduced_action_size = 9
