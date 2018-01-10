#!/bin/python3
__author__ = "Changjian Li"

import numpy as np
from observation import get_vehicle_state, Lanelet_graph, get_veh_next_edge_id, NUM_VEHICLE_CONSIDERED, MAX_ACCELERATION

t_gap = 3

def get_reward(self):
  r_safety = get_reward_safety(self)
#  r_regulation = get_reward_regulation()
#  r_mission = get_reward_mission()
#  r_comfort = get_reward_comfort()
#  return [r_safety, r_regulation, r_mission, r_comfort, r_speed]
  return
  
def get_reward_safety(self):
  veh_state_dict = get_vehicle_state()
  ego_state_dict = veh_state_dict["ego"]
  r = 0
  for veh_id, state_dict in veh_state_dict. items():
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

def get_reward_regulation(env):
  r = []
  obs_dict = get_observation()
  veh_state_dict = get_env_state()
  
  if obs_dict["ego_dist_to_end_of_lane"] < 0.5:
    flag = False
    for i in range(NUM_VEHICLE_CONSIDERED):
      if obs_dict["exists_vehicle"][i] == 1 and
         obs_dict["has_priority"][i] == 1 and 
         obs_dict["lane_relation"][i] <= 2:
        flag = True
    r += [(obs_dict["ego_dist_to_end_of_lane"]-0.5)]
  
  if obs_dict["ego_dist_to_end_of_lane"] = 0.5 + (veh_state_dict["ego"]["speed"] - MAX_ACCELERATION)*1:
    # ************INFORMAL*****************
    if env.action["lane_change"] != 0:
      r += [-1]
  
  flag = False
  for i in range(NUM_VEHICLE_CONSIDERED):
    if obs_dict["exists_vehicle"][i] == 1 and
       obs_dict["lane_relation"][i] == 1 and 
       obs_dict[""][i] <= 2:





#def get_reward_mission():
#  return

def get_reward_comfort():
 

def get_reward_speed(vel, max_vel):
  return vel-max_vel
