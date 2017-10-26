#!/bin/python3
__author__ = "Changjian Li"

import numpy as np
from get_env_state import get_vehicle_state, get_lanelet_graph

t_gap = 3

def get_reward():
  r_safety = get_reward_safety()
#  r_regulation = get_reward_regulation()
#  r_mission = get_reward_mission()
#  r_comfort = get_reward_comfort()
#  return [r_safety, r_regulation, r_mission, r_comfort]
  return
  
def get_reward_safety():
  veh_state_dict = get_vehicle_state()
  ego_state_dict = veh_state_dict["ego"]
  r = 0
  for veh_id, state_dict in veh_state_dict.iteritems():
    if veh_id != "ego":
      r += calc_reward_safety(ego_state_dict["location"], ego_state_dict["angle"], ego_state_dict["velocity"],
                              state_dict["location"], state_dict["angle"], state_dict["velocity"])
  return r

def calc_reward_safety(pos_ego, ang_ego, vel_ego, pos, ang, vel):
  dist = np.linalg.norm(pos_ego - pos)
  ang_ego = ang_ego * np.pi / 180
  ang = ang * np.pi / 180
  # north is the zero for angle, so sine for x and cos for y
  vel_ego = (vel_ego * sin(ang_ego), vel * cos(ang_ego))
  vel = (vel * sin(ang), vel * cos(ang))
  m = dist - np.array(vel_ego - vel)*np.array(pos_ego - pos)*t/dist
  if m > 0:
    return 0
  else:
    return np.exp(m)
  
"""
the regulations are difficult to encode in the form of usual reward, so we can make it rule based and
output a series of acceptable actions, which is compatible with the rest of the learning architecture
"""



def get_reward_mission():
  return
