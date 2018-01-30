#!/bin/python3
__author__ = "Changjian Li"

from include import *

def get_reward_list(env):
  r_safety = get_reward_safety(env)
  r_regulation = get_reward_regulation(env)
  r_comfort = get_reward_comfort(env)
  r_speed = get_reward_speed(env)
  return [r_safety, r_regulation, r_speed, r_comfort]
  
def get_reward_safety(env):
  if env.env_state == EnvState.CRASH:
    return -1
  return 0

def get_reward_regulation(env):
  obs_dict = env.obs_dict_hist.get(-1)
  veh_dict = env.veh_dict_hist.get(-1)
  
  if obs_dict["ego_dist_to_end_of_lane"] < 0.5:
    if obs_dict["ego_correct_lane_gap"] != 0:
      return -1
    
    for i in range(env.NUM_VEHICLE_CONSIDERED):
      if obs_dict["exists_vehicle"][i] == 1 and \
         obs_dict["has_priority"][i] == 1 and \
         (obs_dict["veh_relation_peer"][i] == 1 or obs_dict["veh_relation_conflict"][i] == 1) and \
         obs_dict["dist_to_end_of_lane"][i] < 10 * obs_dict["speed"][i] and \
         obs_dict["ego_speed"] > 0.2:
        return -1

  return 0

def get_reward_comfort(env):
  r = 0
  if env.veh_dict_hist.size < 2:
    return r
  
  if env.veh_dict_hist.get(-1)[env.EGO_VEH_ID]["edge_id"] == env.veh_dict_hist.get(-2)[env.EGO_VEH_ID]["edge_id"] and \
     env.veh_dict_hist.get(-1)[env.EGO_VEH_ID]["lane_id"] != env.veh_dict_hist.get(-2)[env.EGO_VEH_ID]["lane_id"]:
    r += -0.5
  
  accel = (env.veh_dict_hist.get(-1)[env.EGO_VEH_ID]["speed"] - env.veh_dict_hist.get(-2)[env.EGO_VEH_ID]["speed"])/env.SUMO_TIME_STEP
  if accel > 0 and abs(accel) > env.MAX_COMFORT_ACCEL:
    r += -0.5 * (abs(accel) - env.MAX_COMFORT_ACCEL)/(env.MAX_VEH_ACCEL - env.MAX_COMFORT_ACCEL)
  elif accel < 0 and abs(accel) > env.MAX_COMFORT_DECEL:
    r += -0.5 * (abs(accel) - env.MAX_COMFORT_DECEL)/(env.MAX_VEH_DECEL - env.MAX_COMFORT_DECEL)

  return r
  
def get_reward_speed(env):
  ego_max_speed = min(env.tc.vehicle.getAllowedSpeed(env.EGO_VEH_ID), env.MAX_VEH_SPEED)
  if env.veh_dict_hist.get(-1)[env.EGO_VEH_ID]["speed"] < ego_max_speed and \
     env.action_hist.get(-1)["accel_level"].value <= ActionAccel.NOOP.value:
       return -1
  return 0
