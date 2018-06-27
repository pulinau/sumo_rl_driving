#!/bin/python3
__author__ = "Changjian Li"

from include import *

def get_reward_list(env):
  if env.env_state == EnvState.DONE:
    return [0] * 4
  r_validity = None
  r_safety = 10 * get_reward_safety(env)
  r_regulation = 10 * get_reward_regulation(env)
  r_speed_comfort = None
  return [r_validity, r_safety, r_regulation, r_speed_comfort]

def get_reward_safety(env):
  if env.env_state == EnvState.CRASH:
    return -1

  obs_dict = env.obs_dict_hist[-1]
  for i in range(env.NUM_VEH_CONSIDERED):
    if obs_dict["exists_vehicle"][i] == 1:
      ego_v = np.array([0, obs_dict["ego_speed"][i]])
      speed = obs_dict["speed"][i]
      angle = obs_dict["relative_heading"][i] + np.pi / 2
      v = np.array([speed * np.cos(angle), speed * np.sin(angle)])
      if obs_dict["veh_relation_ahead"][i] or \
         obs_dict["veh_relation_behind"][i] or \
         obs_dict["veh_relation_next"][i] or \
         obs_dict["veh_relation_prev"][i] or \
         obs_dict["veh_relation_conflict"][i] or \
         obs_dict["veh_relation_peer"][i]:
        pos = obs_dict["relative_position"][i]
        ttc = np.dot(pos, pos) / max(np.dot((ego_v - v), pos), 0.001)
        if abs(ttc) < 2:
          return -1

  return 0

def get_reward_regulation(env):
  obs_dict = env.obs_dict_hist[-1]
  veh_dict = env.veh_dict_hist[-1]
  
  if obs_dict["ego_dist_to_end_of_lane"] < 2:
    if obs_dict["ego_correct_lane_gap"] != 0:
      return -1
    
    for i in range(env.NUM_VEH_CONSIDERED):
      if obs_dict["exists_vehicle"][i] == 1 and \
         obs_dict["has_priority"][i] == 1 and \
         (obs_dict["veh_relation_peer"][i] == 1 or obs_dict["veh_relation_conflict"][i] == 1) and \
         obs_dict["dist_to_end_of_lane"][i] < 2 and \
         obs_dict["ego_speed"] > 0.2:
        return -1

  return 0

def get_reward_comfort(env):
  r = 0
  if len(env.veh_dict_hist) < 2:
    return r
  
  if (env.veh_dict_hist[-1][env.EGO_VEH_ID]["edge_id"] == env.veh_dict_hist[-2][env.EGO_VEH_ID]["edge_id"] and 
      env.veh_dict_hist[-1][env.EGO_VEH_ID]["lane_id"] != env.veh_dict_hist[-2][env.EGO_VEH_ID]["lane_id"]
      ) or env.env_state == EnvState.CRASH:
    r += -0.5

  ego_max_accel = min(env.tc.vehicle.getAccel(env.EGO_VEH_ID), env.MAX_VEH_ACCEL)
  ego_max_decel = min(env.tc.vehicle.getDecel(env.EGO_VEH_ID), env.MAX_VEH_DECEL)  
  
  accel_level = env.action_dict_hist[-1]["accel_level"]
  #print("accel")
  if accel_level.value > env.MAX_COMFORT_ACCEL_LEVEL.value:
    r += -0.5 * (accel_level.value - env.MAX_COMFORT_ACCEL_LEVEL.value)/(ActionAccel.MAXACCEL.value - env.MAX_COMFORT_ACCEL_LEVEL.value)
  elif accel_level.value < env.MAX_COMFORT_DECEL_LEVEL.value:
    r += -0.5 * (env.MAX_COMFORT_DECEL_LEVEL.value - accel_level.value)/(env.MAX_COMFORT_DECEL_LEVEL.value - ActionAccel.MAXDECEL.value)

  return r
  
def get_reward_speed(env):
  ego_max_speed = min(env.tc.vehicle.getAllowedSpeed(env.EGO_VEH_ID), env.MAX_VEH_SPEED)
  if env.veh_dict_hist[-1][env.EGO_VEH_ID]["speed"] < ego_max_speed and \
     env.action_dict_hist[-1]["accel_level"].value <= ActionAccel.NOOP.value:
       return -1
  return 0
