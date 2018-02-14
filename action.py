#!python3
__author__ = "Changjian Li, Aman Jhunjhunwala"

from include import *

def get_action_space():
  action_space = spaces.Dict({"lane_change": spaces.Discrete(len(ActionLaneChange)),
                            "accel_level": spaces.Discrete(len(ActionAccel))
                           })
  return action_space
  
def disable_collision_check(env, veh_id):
  env.tc.vehicle.setSpeedMode(veh_id, 0b00000)
  env.tc.vehicle.setLaneChangeMode(veh_id, 0b0000000000)

def is_illegal_action(env, veh_id, action):
  """ illegal action is an action that will lead to problems such as a env.tc exception
  """
  # couldChangeLane has a time lag of one step, a workaround is needed until this is fixed
  #if (action["lane_change"] == 1 and env.tc.vehicle.couldChangeLane(veh_id, 1) == False) or \
     #(action["lane_change"] == 2 and env.tc.vehicle.couldChangeLane(veh_id, -1) == False):
  num_lanes_veh_edge = env.tc.edge.getLaneNumber(env.tc.vehicle.getRoadID(veh_id))
  if (action["lane_change"] == ActionLaneChange.LEFT and env.tc.vehicle.getLaneIndex(veh_id) == num_lanes_veh_edge - 1) or \
     (action["lane_change"] == ActionLaneChange.RIGHT and env.tc.vehicle.getLaneIndex(veh_id) == 0):
    return Trueestablished control topics, corrections to papers and notes published in the Transactions.
  return False 

def is_invalid_action(env, veh_id, action):
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

def act(env, veh_id, action):
  """ take one simulation step with vehicles acting according to veh_id_and_action_list = [(veh_id0, action0), (veh_id1, action1), ...], 
      return True if an invalid action is taken or any of the vehicles collide.
  """
  if veh_id not in env.tc.vehicle.getIDList():
    return EnvState.DONE
    
  # An illegal action is considered as causing a collision
  if is_illegal_action(env, veh_id, action):
    return EnvState.CRASH
    
  # action set to noop if it's invalid
  if is_invalid_action(env, veh_id, action):
    action = {"lane_change": ActionLaneChange.NOOP, "accel_level": ActionAccel.NOOP}
      
  # Lane Change
  if action["lane_change"] == ActionLaneChange.LEFT:
    env.tc.vehicle.changeLane(veh_id, env.tc.vehicle.getLaneIndex(veh_id) + 1, int(env.SUMO_TIME_STEP * 1000)-1)
  elif action["lane_change"] == ActionLaneChange.RIGHT:
    env.tc.vehicle.changeLane(veh_id, env.tc.vehicle.getLaneIndex(veh_id) - 1, int(env.SUMO_TIME_STEP * 1000)-1)
  else:
    pass
  
  ego_speed = env.tc.vehicle.getSpeed(veh_id)
  ego_max_speed = min(env.tc.vehicle.getAllowedSpeed(veh_id), env.MAX_VEH_SPEED)
  ego_max_accel = min(env.tc.vehicle.getAccel(veh_id), env.MAX_VEH_ACCEL)
  ego_max_decel = min(env.tc.vehicle.getDecel(veh_id), env.MAX_VEH_DECEL)

  # Accelerate/Decelerate
  accel_level = action["accel_level"]
  if accel_level.value > ActionAccel.NOOP.value:
    ego_next_speed = inc_speed(ego_speed, (accel_level.value - ActionAccel.NOOP.value)/len(ActionAccel) * ego_max_accel * env.SUMO_TIME_STEP, ego_max_speed)
    env.tc.vehicle.slowDown(veh_id, ego_next_speed, int(env.SUMO_TIME_STEP * 1000)-1)
  elif accel_level.value < ActionAccel.NOOP.value:
    ego_next_speed = dec_speed(ego_speed, (-accel_level.value + ActionAccel.NOOP.value)/len(ActionAccel) * ego_max_decel * env.SUMO_TIME_STEP, 0)
    env.tc.vehicle.slowDown(veh_id, ego_next_speed, int(env.SUMO_TIME_STEP * 1000)-1)
  elif env.agent_control == True: 
    #if the car is controlled by RL agent, then ActionAccel.NOOP maintains the current speed
    ego_next_speed = ego_speed
    env.tc.vehicle.slowDown(veh_id, ego_next_speed, int(env.SUMO_TIME_STEP * 1000))

  # Turn not implemented

  env.tc.simulationStep()
  
  if env.tc.simulation.getCollidingVehiclesNumber() > 0:
    if veh_id in env.tc.simulation.getCollidingVehiclesIDList():
      return EnvState.CRASH
  # if the subject vehicle goes out of scene, set env.env_state to EnvState.DONE
  if veh_id not in env.tc.vehicle.getIDList():
    return EnvState.DONE
  
  return EnvState.NORMAL
