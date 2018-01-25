#!python3
__author__ = "Changjian Li"

from copy import deepcopy
import random
import inspect

from action import get_action_space, disable_collision_check, act, EnvState
from observation import get_observation_space, get_veh_dict, get_obs_dict
from reward import get_reward_list

from include import *

class History:
  def __init__(self, length):
    self.history = []
    self.length = length
    
  def add(self, current):
    if len(self.history) >= self.length:
      self.history = self.history[:-1]
    self.history += [current]

  def reset(self):
    self.history = []

  def get(self, index):
    return deepcopy(self.history[index])

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}

class SumoGymEnv(gym.Env):
  """SUMO environment"""
  def __init__(self, config):
    _attrs = class_vars(config)
    for _attr in _attrs:
      setattr(self, _attr, getattr(config, _attr))
    
    self.action_space = get_action_space()
    self.obsevation_space = get_observation_space(self)
    
    self.env_state = EnvState.NOT_STARTED
    self.veh_dict_hist = History(2)
    self.obs_dict_hist = History(2)
    self.action_hist = History(2)
    
    sim_label = "sim" + str(random.randint(0, 65536))
    traci.start(self.SUMO_CMD, label = sim_label)
    self.tc = traci.getConnection(sim_label) 
    self.tc.simulationStep()
    disable_collision_check(self, self.EGO_VEH_ID)
    pass

  def _step(self, action):
    obs = 0
    done = 0
    reward = 0
    info = None
    return obs, reward, done, info

  def _reset(self):
    self.tc.close()
    self.env_state = EnvState.NOT_STARTED
    self.action_hist.reset()
    self.veh_dict_hist.reset()
    self.obs_dict_hist.reset()
    sim_label = "sim" + str(random.randint(0, 65536))
    traci.start(self.SUMO_CMD, label = sim_label)
    self.tc = traci.getConnection(sim_label) 

  def _close(self):
    self.tc.close()
    
class MultiObjSumoEnv(SumoGymEnv):
  def step(self, action):
    self.env_state = act(self, self.EGO_VEH_ID, action)
    obs_dict =  get_obs_dict(self)
    self.action_hist.add(action)
    self.veh_dict_hist.add(get_veh_dict(self))
    self.obs_dict_hist.add(obs_dict)
    info = None
    # return get_obs_and_rewards_list(self), self.done, info
