#!/bin/python3
__author__ = "Changjian Li"

from include import *
import numpy as np

def get_reward_list(env):
  r_validity, d_validity = None, None
  r_safety, d_safety = get_reward_safety(env)
  r_regulation, d_regulation = get_reward_regulation(env)
  r_speed_comfort, d_speed_comfort = None, None
  return ([r_validity, r_safety, r_regulation, r_speed_comfort], [d_validity, d_safety, d_regulation, d_speed_comfort])

def get_reward_safety(env):
  rewards = []
  dones = []
  obs_dict = env.obs_dict_hist[-1]
  old_obs_dict = None
  if len(env.obs_dict_hist) > 1:
    old_obs_dict = env.obs_dict_hist[-2]
  action_dict = env.action_dict_hist[-1]

  for i, c in enumerate(obs_dict["collision"]):
    r = 0
    d = False
    if (old_obs_dict is not None and (
          old_obs_dict["ttc"][i] > obs_dict["ttc"][i] + 0.0000001 or (
            np.linalg.norm(old_obs_dict["relative_position"][i]) > np.linalg.norm(obs_dict["relative_position"][i]) + 0.001 and
            np.linalg.norm(obs_dict["relative_position"][i]) <  4)
          ) and
        (obs_dict["ttc"][i] < 4 or np.linalg.norm(obs_dict["relative_position"][i]) < 8)
        ) or (env.env_state == EnvState.CRASH and c == 1
        ) or (action_dict["lane_change"] != ActionAccel.NOOP and obs_dict["ttc"][i] < 4):
      r = -1
    if obs_dict["is_new"][i] == 1 or r == -1:
      d = True
    rewards += [[r]]
    dones += [[d]]

  return (rewards, dones)

def get_reward_regulation(env):
  r = 0
  done = False

  obs_dict = env.obs_dict_hist[-1]
  old_obs_dict = None
  if len(env.obs_dict_hist) > 1:
    old_obs_dict = env.obs_dict_hist[-2]

  if obs_dict["ego_dist_to_end_of_lane"] < 100:
    if obs_dict["ego_correct_lane_gap"] != 0:
      r = 1/(1 + np.exp(-0.1*(obs_dict["ego_dist_to_end_of_lane"]-20))) - 1

  if obs_dict["ego_dist_to_end_of_lane"] < 3 and \
     obs_dict["ego_has_priority"] != 1 and \
     obs_dict["ego_in_intersection"] != 1 and \
     old_obs_dict["ego_dist_to_end_of_lane"] > obs_dict["ego_dist_to_end_of_lane"] + 0.01:
      r = -1

  if obs_dict["ego_priority_changed"] == 1 or obs_dict["ego_edge_changed"] == 1:
    done = True

  return ([[r]], [[done]])
