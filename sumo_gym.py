#!python3

SUMO_TOOLS_DIR = "/home/ken/project/sumo-0.30.0/tools"
SUMO_BIN = "/home/ken/project/sumo-0.30.0/bin/sumo"
SUMO_CONFIG = "/home/ken/project/sumo/test.sumocfg"
SUMO_CMD = [SUMO_BIN, "-c", SUMO_CONFIG, "--time-to-teleport", "-1"]

OCCUPANCY_GRID_NUM_RING = 10
OCCUPANCY_GRID_NUM_GRID_PER_RING = 8
OCCUPANCY_GRID_NUM_GRID = OCCUPANCY_GRID_NUM_GRID_PER_RING * OCCUPANCY_GRID_NUM_RING

import gym
from gym import spaces
import numpy as np
import sys
try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import time
import traci


Class SumoEnv(gym.Env):
  """
  action_space is a spaces.Dict Object:
    1) "lane_change": Discrete(3) - NOOP[0], LEFT[1], RIGHT[2]
    2) "turn": Discrete(3) - NOOP[0], LEFT[1], RIGHT[2]
    3) "speed_level": Discrete(8) - BRAKEHARD[0], BRAKE[1], SPEED1-5[2-6], MAXSPEED[7]
  object_space is an occupancy grid of the environment centred at the ego vehicle
  """
  def __init__(self):
    self.action_space = spaces.Dict({"lane_change": spaces.Discrete(3), 
                                     "turn": spaces.Discrete(3),
                                     "speed_level": spaces.Discrete(8)
                                     })
    self.obsevation_space = spaces.Dict({
                                         "vehicle_0": spaces.MultiDiscrete([0, 1] * OCCUPANCY_GRID_NUM_RING),
                                         "vehicle_1": spaces.MultiDiscrete([0, 1] * OCCUPANCY_GRID_NUM_RING),
                                         "vehicle_2": spaces.MultiDiscrete([0, 1] * OCCUPANCY_GRID_NUM_RING),
                                         "pedestrian_0": spaces.MultiDiscrete([0, 1] * OCCUPANCY_GRID_NUM_RING),
                                         "pedestrian_1": spaces.MultiDiscrete([0, 1] * OCCUPANCY_GRID_NUM_RING),
                                         "pedestrian_2": spaces.MultiDiscrete([0, 1] * OCCUPANCY_GRID_NUM_RING),
                                        })
    traci.start(SUMO_CMD)
    pass
  def _step(self, action):
    traci.simulationStep()
    pass
  def _reset(self):
    traci.close()
    traci.start(SUMO_CMD)
    pass
  def _close(self):
    traci.colse()