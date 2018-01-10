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

from observation import get_edge_dict

class ActionLaneChange(Enum):
  NOOP = 0
  LEFT = 1
  RIGHT = 2

class ActionAccel(Enum):
  MAXDECEL = 0
  MEDDECEL = 1
  MINDECEL = 2
  NOOP = 3
  MINACCEL = 4
  MEDACCEL = 5
  MAXDECEL = 6
  
action_space = spaces.Dict({"lane_change": spaces.Discrete(len(ActionLaneChange)),
                            "accel_level": spaces.Discrete(len(ActionAccel))
                           })
  
def disable_collision_check(veh_id):
  traci.vehicle.setSpeedMode(veh_id, 0b00000)
  traci.vehicle.setLaneChangeMode(veh_id, 0b0000000000)

def is_illegal_action(veh_id, action):
  """ illegal action is an action that will lead to problems such as a traci exception
  """
  # couldChangeLane has a time lag of one step, a workaround is needed until this is fixed
  #if (action["lane_change"] == 1 and traci.vehicle.couldChangeLane(veh_id, 1) == False) or \
     #(action["lane_change"] == 2 and traci.vehicle.couldChangeLane(veh_id, -1) == False):
  num_lanes_veh_edge = traci.edge.getLaneNumber(traci.vehicle.getRoadID(veh_id))
  if (action["lane_change"] == ActionLaneChange.LEFT and traci.vehicle.getLaneIndex(veh_id) == num_lanes_veh_edge - 1) or \
     (action["lane_change"] == ActionLaneChange.RIGHT and traci.vehicle.getLaneIndex(veh_id) == 0):
    return True
  return False 

def is_invalid_action(veh_id, action):
  """ invalid action is an action that doesn't make sense, it's treated as a noop
  """
  return False

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

def act(self, veh_id, action):
  """ take one simulation step with vehicles acting according to veh_id_and_action_list = [(veh_id0, action0), (veh_id1, action1), ...], 
      return True if an invalid action is taken or any of the vehicles collide.
  """
  if self.done:
    return True
    
  # An illegal action is considered as causing a collision
  if is_illegal_action(veh_id, action):
    return True
    
  # action set to noop if it's invalid
  if is_invalid_action(veh_id, action):
    action = {"lane_change": ActionLaneChange.NOOP, "accel_level": ActionAccel.NOOP}
      
  # Lane Change
  if action["lane_change"] == ActionLaneChange.LEFT:
    traci.vehicle.changeLane(veh_id, traci.vehicle.getLaneIndex(veh_id) + 1, int(SUMO_TIME_STEP * 1000)-1)
  elif action["lane_change"] == ActionLaneChange.RIGHT:
    traci.vehicle.changeLane(veh_id, traci.vehicle.getLaneIndex(veh_id) - 1, int(SUMO_TIME_STEP * 1000)-1)
  else:
    pass
  
  ego_speed = traci.vehicle.getSpeed(veh_id)
  ego_max_speed = min(traci.vehicle.getAllowedSpeed(veh_id), MAX_VEH_SPEED)
  ego_max_accel = min(traci.vehicle.getAccel(veh_id), MAX_VEH_ACCEL)
  ego_max_decel = min(traci.vehicle.getDecel(veh_id), MAX_VEH_DECEL)

  # Accelerate/Decelerate
  accel_level = action["accel_level"]
  if accel_level > ActionAccel.NOOP:
    ego_next_speed = inc_speed(ego_speed, (accel_level - ActionAccel.NOOP)/len(ActionAccel) * ego_max_accel * SUMO_TIME_STEP, ego_max_speed)
    traci.vehicle.slowDown(veh_id, ego_next_speed, int(SUMO_TIME_STEP * 1000)-1)
  elif accel_level < ActionAccel.NOOP:
    ego_next_speed = dec_speed(ego_speed, (-accel_level + ActionAccel.NOOP)/len(ActionAccel) * ego_max_decel * SUMO_TIME_STEP, 0)
    traci.vehicle.slowDown(veh_id, ego_next_speed, int(SUMO_TIME_STEP * 1000)-1)
  else:
    ego_next_speed = ego_speed
    traci.vehicle.slowDown(veh_id, ego_next_speed, int(SUMO_TIME_STEP * 1000))

  # Turn not implemented

  traci.simulationStep()
  
  if traci.simulation.getCollidingVehiclesNumber() > 0:
    if veh_id in traci.simulation.getCollidingVehiclesIDList():
      return True
  
  return False
