#!python3
__author__ = "Changjian Li, Aman Jhunjhunwala"

import numpy as np
import sys
import time

import gym
from gym import spaces

from parameters import *
try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import traci

action_space = spaces.Dict({"lane_change": spaces.Discrete(3),
                            "accel_level": spaces.Discrete(7)
                           })

def check_action(action):
  assert (action["lane_change"] == 0 or
          (action["lane_change"] == 1 and traci.vehicle.couldChangeLane(EGO_VEH_ID, 1) == True) or
          (action["lane_change"] == 2 and traci.vehicle.couldChangeLane(EGO_VEH_ID, 0) == True)
          ), "invalid action"

def inc_speed(speed, inc, max_speed):
  if (speed + inc) > max_speed:
    return max_speed
  else:
    return speed + inc

def dec_speed(speed, dec, min_speed):
  if (speed - dec) < min_speed:
    return min_speed
  else:
    return speed - dec

def act(action):
  assert traci.simulation.getCollisionNumber() > 0, "collision already occurred"
  
  check_action()
    
  # Lane Change
  if action["lane_change"] == 1:
    traci.vehicle.changeLane(EGO_VEH_ID, traci.vehicle.getLaneIndex(ego_car_id) + 1, SUMO_TIME_STEP)
  elif action["lane_change"] == 2:
    traci.vehicle.changeLane(EGO_VEH_ID, traci.vehicle.getLaneIndex(ego_car_id) - 1, SUMO_TIME_STEP)
  else:
    pass
  
  ego_speed = traci.vehicle.getSpeed(EGO_VEH_ID)
  ego_max_speed = min(traci.vehicle.getAllowedSpeed(EGO_VEH_ID), MAX_VEH_SPEED)
  ego_max_accel = min(traci.vehicle.getAccel(EGO_VEH_ID), MAX_VEH_ACCEL)
  ego_max_decel = min(traci.vehicle.getDecel(EGO_VEH_ID), MAX_VEH_DECEL)

  # Accelerate/Decelerate
  accel_level = action["accel_level"]
  if accel_level > 3:
    ego_next_speed = inc_speed(ego_speed, (accel_level - 3)/3 * ego_max_accel * SUMO_TIME_STEP, ego_max_speed)
    traci.vehicle.slowdown(ego_next_speed, SUMO_TIME_STEP)
  elif accel_level < 3:
    ego_next_speed = dec_speed(ego_speed, (-accel_level + 3)/3 * ego_max_decel * SUMO_TIME_STEP, 0)
    traci.vehicle.slowdown(ego_next_speed, SUMO_TIME_STEP)
  else:
    pass

  # Turn not implemented

  traci.simulationStep()
  if traci.simulation.getCollisionNumber() > 0:
    collision = True
  else:
    collision = False
  
  return collision
