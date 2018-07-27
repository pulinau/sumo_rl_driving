#!/bin/python3
__author__ = "Changjian Li"

from include import *
import numpy as np

def get_reward_list(env):
  if env.env_state == EnvState.DONE:
    return [0] * 5
  r_validity = None
  r_safety = get_reward_safety(env)
  r_regulation = get_reward_regulation(env)
  r_speed_comfort = None
  return [r_validity, r_safety, r_regulation, r_speed_comfort]

def get_reward_safety(env):
  rewards = []
  obs_dict = env.obs_dict_hist[-1]

  for i, c in enumerate(obs_dict["collision"]):
    r = 0
    if (env.env_state == EnvState.CRASH and c == True):
      r = -1
    rewards += [[r]]

  return rewards

def get_reward_regulation(env):
  obs_dict = env.obs_dict_hist[-1]

  r = 0
  if obs_dict["ego_dist_to_end_of_lane"] < 5:
    if obs_dict["ego_correct_lane_gap"] != 0:
      r = -1

    for i in range(env.NUM_VEH_CONSIDERED):
      if obs_dict["exists_vehicle"][i] == 1 and \
         obs_dict["has_priority"][i] == 1 and \
         (obs_dict["veh_relation_peer"][i] == 1 or obs_dict["veh_relation_conflict"][i] == 1) and \
         obs_dict["dist_to_end_of_lane"][i] < 60 and \
         obs_dict["ego_speed"] > 0.4:
        r = -1

  return [[r]]
