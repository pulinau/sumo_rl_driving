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

from action import action_space, act
from observation import observation_space, get_obs

class SumoGymEnv(gym.Env):
  """SUMO environment"""
  def __init__(self):
    self.action_space = action_space
    # observation is not used anyway, so it's arbitrarily set
    self.obsevation_space = observation_space
    
    traci.start(self.SUMO_CMD)
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
    traci.colse()
    
Class MultiObjSumoEnv(SumoGymEnv):
  def step(self, action):
    obs_list = get_obs()
    done = act(action)
    reward_list = get_reward()
    assert len(obs_list) == len(reward_list)
    info = None
    return obs_list, reward_list, done, info
