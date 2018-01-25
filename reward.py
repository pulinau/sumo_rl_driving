#!/bin/python3
__author__ = "Changjian Li"



def get_reward_list(env):
  r_safety = get_reward_safety(env)
  r_regulation = get_reward_regulation()
  r_comfort = get_reward_comfort()
  r_speed = get_reward_speed()
  return [r_safety, r_regulation, r_comfort, r_speed]
  
def get_reward_safety(env):
  if env.state == EnvState.CRASH:
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
         (obs_dict["lane_relation_peer"][i] == 1 or obs_dict["lane_relation_conflict"][i] == 1):
        return -1

  return 0

def get_reward_comfort(env):
  if env.veh_dict_hist.get(-1)[env.EGO_VEH_ID]["edge_id"] == env.veh_dict_hist.get(-2)[env.EGO_VEH_ID]["edge_id"] and \
     env.veh_dict_hist.get(-1)[env.EGO_VEH_ID]["lane_id"] != env.veh_dict_hist.get(-2)[env.EGO_VEH_ID]["lane_id"]:
    return -1
  
  accel = (env.veh_dict_hist.get(-1)[env.EGO_VEH_ID]["speed"] - env.veh_dict_hist.get(-2)[env.EGO_VEH_ID]["speed"])/env.SUMO_TIME_STEP
  if accel > 0:
    if abs(accel) <= env.MAX_COMFORT_ACCEL:
      return 0
    else:
      return -(abs(accel) - env.MAX_COMFORT_ACCEL)/(env.MAX_VEH_ACCEL - env.MAX_COMFORT_ACCEL)
  else:
    if abs(accel) <= env.MAX_COMFORT_DECEL:
      return 0
    else:
      return -(abs(accel) - env.MAX_COMFORT_DECEL)/(env.MAX_VEH_DECEL - env.MAX_COMFORT_DECEL)

def get_reward_speed(env):
  ego_max_speed = min(env.tc.vehicle.getAllowedSpeed(env.EGO_VEH_ID), env.MAX_VEH_SPEED)
  return (env.ego_dict_hist.get(-1)["speed"] - ego_max_speed) / ego_max_speed
