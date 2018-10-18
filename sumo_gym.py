#!python3
__author__ = "Changjian Li"

import random

from action import get_action_space, disable_collision_check, enable_collision_check, act, infer_action
from observation import get_observation_space, get_veh_dict, get_obs_dict, get_lanelet_dict, get_edge_dict
from reward import get_reward_list
from utils import class_vars
from collections import deque
from copy import deepcopy

from include import *

class SumoCfg():
  def __init__(self, 
               # sumo
               SUMO_CMD, 
               SUMO_TIME_STEP, 
               NET_XML_FILE,
               ROU_XML_FILE_LIST,
               EGO_VEH_ID, 
               MAX_VEH_ACCEL, 
               MAX_VEH_DECEL, 
               MAX_VEH_SPEED, 
               # observation
               NUM_LANE_CONSIDERED, 
               NUM_VEH_CONSIDERED,
               MAX_TTC_CONSIDERED,
               OBSERVATION_RADIUS, 
               # reward
               MAX_COMFORT_ACCEL_LEVEL, 
               MAX_COMFORT_DECEL_LEVEL,
               # color
               DEFAULT_COLOR,
               YIELD_COLOR):
    self.SUMO_CMD = SUMO_CMD
    self.SUMO_TIME_STEP = SUMO_TIME_STEP
    self.NET_XML_FILE = NET_XML_FILE
    self.ROU_XML_FILE_LIST = ROU_XML_FILE_LIST
    self.EGO_VEH_ID = EGO_VEH_ID
    self.MAX_VEH_ACCEL = MAX_VEH_ACCEL
    self.MAX_VEH_DECEL = MAX_VEH_DECEL
    self.MAX_VEH_SPEED = MAX_VEH_SPEED
    
    self.NUM_LANE_CONSIDERED = NUM_LANE_CONSIDERED
    self.NUM_VEH_CONSIDERED = NUM_VEH_CONSIDERED
    self.MAX_TTC_CONSIDERED = MAX_TTC_CONSIDERED
    self.OBSERVATION_RADIUS = OBSERVATION_RADIUS
    
    self.MAX_COMFORT_ACCEL_LEVEL = MAX_COMFORT_ACCEL_LEVEL
    self.MAX_COMFORT_DECEL_LEVEL = MAX_COMFORT_DECEL_LEVEL

    self.DEFAULT_COLOR = DEFAULT_COLOR
    self.YIELD_COLOR = YIELD_COLOR

class SumoGymEnv(gym.Env):
  """SUMO environment"""
  def __init__(self, config):
    _attrs = class_vars(config)
    for _attr in _attrs:
      setattr(self, _attr, getattr(config, _attr))
    
    self.action_space = get_action_space()
    self.obsevation_space = get_observation_space(self)

    self.lanelet_dict = get_lanelet_dict(self.NET_XML_FILE)
    self.edge_dict = get_edge_dict(self.NET_XML_FILE)

    self.env_state = EnvState.NOT_STARTED
    self._agt_ctrl = False # whether the ego car is controlled by RL agent
    self.veh_dict_hist = deque(maxlen=2)
    self.obs_dict_hist = deque(maxlen=2)
    self.action_dict_hist = deque(maxlen=2)

    self.sim_label = "sim" + str(random.randint(0, 65536))
    ROU_XML_FILE = random.sample(self.ROU_XML_FILE_LIST, 1)
    traci.start(self.SUMO_CMD + ROU_XML_FILE, label=self.sim_label)
    self.tc = traci.getConnection(self.sim_label)

  @property
  def agt_ctrl(self):
    return self._agt_ctrl
  @agt_ctrl.setter
  def agt_ctrl(self, value):
    if value == True:
      disable_collision_check(self, self.EGO_VEH_ID)
      self._agt_ctrl = value
    elif value == False:
      enable_collision_check(self, self.EGO_VEH_ID)
      self._agt_ctrl = value
    else:
      raise ValueError("SumoGymEnv.agt_ctrl must be either True or False")
      
  def step(self, action):
    obs = 0
    done = 0
    reward = 0
    info = None
    return obs, reward, done, info

  def reset(self, init_step):
    self.action_dict_hist.clear()
    self.veh_dict_hist.clear()
    self.obs_dict_hist.clear()
    try:
      ROU_XML_FILE = random.sample(self.ROU_XML_FILE_LIST, 1)
      self.tc.load(self.SUMO_CMD[1:] + ROU_XML_FILE)
      # 1st time step starts the simulation, 
      # 2nd makes sure that all initial vehicles (departure time < SUMO_TIME_STEP) are in scene
      self.agt_ctrl = False
      self.tc.simulationStep()
      self.tc.simulationStep()
      self.veh_dict_hist.append(get_veh_dict(self))
      self.obs_dict_hist.append(get_obs_dict(self))
      self.env_state = EnvState.NORMAL
      for i in range(init_step):
        if self.env_state == EnvState.NORMAL:
          self.step()
      if self.env_state != EnvState.NORMAL:
        return self.reset(i-1)
      self.agt_ctrl = True
      return get_obs_dict(self)
    except (traci.FatalTraCIError, traci.TraCIException):
      self.env_state = EnvState.ERROR
      raise

  def close(self):
    try:  
      self.tc.close()
    except (traci.FatalTraCIError, traci.TraCIException):
      self.env_state = EnvState.ERROR
      raise
    
class MultiObjSumoEnv(SumoGymEnv):
  def step(self, action_dict=None):
    assert self.env_state == EnvState.NORMAL, "env.env_state is not EnvState.NORMAL"
    try:
      self.env_state = act(self, self.EGO_VEH_ID, action_dict)

      # if ego reaches the end of an incorrect (turning) lane, simulation is considered as DONE
      if self.env_state == EnvState.NORMAL and \
         self.obs_dict_hist[-1]["ego_dist_to_end_of_lane"] < 0.01 and \
         self.obs_dict_hist[-1]["ego_correct_lane_gap"] != 0:
        self.env_state = EnvState.DONE

      if self.env_state == EnvState.DONE:
        obs_dict = deepcopy(self.obs_dict_hist[-1])
        veh_dict = deepcopy(self.veh_dict_hist[-1])
      else:
        obs_dict = get_obs_dict(self)
        veh_dict = get_veh_dict(self)
      self.veh_dict_hist.append(veh_dict)
      self.obs_dict_hist.append(obs_dict)
      if self.agt_ctrl == False:
        action_dict = infer_action(self)
      self.action_dict_hist.append(action_dict)
      info = action_dict
      """
      print(self.obs_dict_hist[-1]["veh_ids"])
      print("peer", self.obs_dict_hist[-1]["veh_relation_peer"])
      print("conflict", self.obs_dict_hist[-1]["veh_relation_conflict"])
      print("ahead", self.obs_dict_hist[-1]["veh_relation_ahead"])
      print("next", self.obs_dict_hist[-1]["veh_relation_next"])
      print("in_intersection", self.obs_dict_hist[-1]["in_intersection"])
      print("has_priority", self.obs_dict_hist[-1]["has_priority"])
      print("ego_has_priority", self.obs_dict_hist[-1]["ego_has_priority"])
      print(self.obs_dict_hist[-1]["ttc"])
      print(get_reward_list(self)[0][1])
      """
      return obs_dict, get_reward_list(self), self.env_state, info
    except (traci.FatalTraCIError, traci.TraCIException):
      self.env_state = EnvState.ERROR
      raise
