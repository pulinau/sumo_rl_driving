#!python3
__author__ = "Changjian Li"

import numpy as np
import sys
import time
from math import pi
import re

import gym
from gym import spaces

from parameters import *
try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import traci

from action import action_space, disable_collision_check, act
from observation import observation_space, get_obs_dict
from reward import get_reward

class SumoGymEnv(gym.Env):
  """SUMO environment"""
  def __init__(self):
    self.action_space = action_space
    self.obsevation_space = observation_space
    
    traci.start(SUMO_CMD)
    traci.simulationStep()
    disable_collision_check(EGO_VEH_ID)
    pass

  def _step(self, action):
    obs = 0
    done = 0
    reward = 0
    info = None
    return obs, reward, done, info

  def _reset(self):
    traci.close()
    traci.start(self.SUMO_CMD)

  def _close(self):
    traci.close()
    
class MultiObjSumoEnv(SumoGymEnv):
  def step(self, action):
    self.done = act(self, EGO_VEH_ID, action)
    self.obs_dict =  get_obs_dict()
    info = None
    #return get_obs_and_rewards_list(self), self.done, info
