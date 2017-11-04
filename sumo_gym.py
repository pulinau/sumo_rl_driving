#!python3

SUMO_TOOLS_DIR = "/home/ken/project/sumo-0.30.0/tools"
SUMO_BIN = "/home/ken/project/sumo-0.30.0/bin/sumo"
SUMO_CONFIG = "/home/ken/project/sumo/test.sumocfg"
SUMO_CMD = [SUMO_BIN, "-c", SUMO_CONFIG, "--time-to-teleport", "-1"]

#OCCUPANCY_GRID_NUM_RING = 10
#OCCUPANCY_GRID_NUM_GRID_PER_RING = 8
#OCCUPANCY_GRID_NUM_GRID = OCCUPANCY_GRID_NUM_GRID_PER_RING * OCCUPANCY_GRID_NUM_RING

MAX_VEHICLE_SPEED = 200
MAX_LANE_PRIORITY = 2
NUM_VEHICLE_CONSIDERED = 16

import gym
from gym import spaces
import numpy as np
import sys
from math import pi

try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import time
import traci

Class InvalidAction(Exception):
  pass

Class SumoEnv(gym.Env):
  """
  action_space is a spaces.Dict object:
    1) "lane_change": Discrete(3) - NOOP[0], LEFT[1], RIGHT[2]
    2) "accel_level": Discrete(7) - DECCEL[0-2], MAINTAIN[3], ACCEL[4-6]
    #3) "turn": Discrete(3) - NOOP[0], LEFT[1], RIGHT[2]
  Some (combination of) actions are not always valid, namely:
    1) LEFT/RIGHT turn with LEFT/RIGHT lane change is invalid, because LEFT/RIGHT turn is always come together with
       lane change, and in the case of intersections, it's not clear what it means by turn-lane_change combination
    2) ACCEL at maximum speed is treated as an noop, same for DECCEL at zero speed (currently the car cannot reverse)
  These are treated as NOOP|MAINTAIN|NOOP
  
  observation space is a spaces.Dict object, whose key is mostly self explanatory:
    1) "ego_speed"
    2) "ego_dist_to_end_of_lane"
    3) "exists_vehicle": spaces.MultiDiscrete object, each of whose element indicates whether a vehicle exists[1] or not[0]
    4) "relative_position": space.Box object of size (NUM_VEHICLE_CONSIDERED, 2), each the position of the vehicle relative to the ego vehicle 
    5) "dist_to_end_of_lane"
    6) "lane_relation": the relationship between the vehicle lane and the ego lane. Possible reltionships are:
         PEER[0]: share the same NEXT lane with the ego lane
         CONFLICT[1]: share the same intersection with the ego lane, and its route conflict that of the ego route
         CONFLICT_INTERSECTION[2]: the lane is inside the intersection ego is approaching, and conflicts with ego route
         NEXT[3]: the next lane of ego lane
         PREV[4]: the previous lane of ego lane
         LEFT[5]: to the left of ego lane
         RIGHT[6]: to the right of ego lane
         IRRELEVANT[7]: the lane is irrelevant to ego vehicle, e.g. opposite lane, in or approaching other intersections, 
                      or in the same intersection but with no conflict with ego route 
  If we feed the observation space to a neural network, the weights connecting to the same key (of the observation space dictionary) should be identical
  If the value of "exists_vehicle" is zero for index i, then the value of all other keys at index i are set to zero
  The weight connection to key "exists_vehicle" should cancel out the effect of other keys, 
  so that if a vehicle doesn't exists at index i, then weighted sum for index i is zero 
  #object_space is an occupancy grid of the environment centred at the ego vehicle
  """
  def __init__(self):
# TO DO: let lanelet information further restrict action space
    self.action_space = spaces.Dict({"lane_change": spaces.Discrete(3), 
                                     #disable turn for now                                    
                                     "speed_level": spaces.Discrete(7)
                                     #"turn": spaces.Discrete(3),
                                     })
    self.obsevation_space = spaces.Dict({"ego_speed": spaces.Box(0, MAX_VEHICLE_SPEED),
                                         "ego_dist_to_end_of_lane": spaces.Box(0, float("inf")),
                                         "ego_in_intersection": spaces.Discrete(2)
                                         "exists_vehicle": spaces.MultiDiscrete([[0,1]] * NUM_VEHICLE_CONSIDERED)
                                         "relative_position": spaces.Box(float("-inf"), float("inf"), (NUM_VEHICLE_CONSIDERED, 2)),
                                         "relative_heading": spaces.Box(-pi, pi, (NUM_VEHICLE_CONSIDERED,)),
                                         "dist_to_end_of_lane": spaces.Box(0, float("inf"), (NUM_VEHICLE_CONSIDERED,)),
                                         "lane_relation": spaces.MultiDiscrete([[0,7]] * NUM_VEHICLE_CONSIDERED)
                                         })
    #self.obsevation_space = spaces.Dict({
                                         ##TO DO: define MultiBox
                                         #"is_intersection": spaces.MultiDiscrete([[0, 1]] * OCCUPANCY_GRID_NUM_RING),
                                         #"lane_priority": spaces.MultiDiscrete([[0, MAX_LANE_PRIORITY]] * OCCUPANCY_GRID_NUM_RING),
                                         #"exists_vehicle_0": spaces.MultiDiscrete([[0, 1]] * OCCUPANCY_GRID_NUM_RING),
                                         #"exists_vehicle_1": spaces.MultiDiscrete([[0, 1]] * OCCUPANCY_GRID_NUM_RING),
                                         #"exists_vehicle_2": spaces.MultiDiscrete([[0, 1]] * OCCUPANCY_GRID_NUM_RING),
                                         #"heading_vehicle_0": MultiBox([[-pi, pi]] * OCCUPANCY_GRID_NUM_RING),
                                         #"heading_vehicle_1": MultiBox([[-pi, pi]] * OCCUPANCY_GRID_NUM_RING),
                                         #"heading_vehicle_2": MultiBox([[-pi, pi]] * OCCUPANCY_GRID_NUM_RING),
                                         #"speed_vehicle_0": MultiBox([[0, MAX_VEHICLE_SPEED]] * OCCUPANCY_GRID_NUM_RING),
                                         #"speed_vehicle_1": MultiBox([[0, MAX_VEHICLE_SPEED]] * OCCUPANCY_GRID_NUM_RING),
                                         #"speed_vehicle_2": MultiBox([[0, MAX_VEHICLE_SPEED]] * OCCUPANCY_GRID_NUM_RING),
                                         ##don't consider pedestrians for now
                                         ##"exists_pedestrian_0": spaces.MultiDiscrete([[0, 1]] * OCCUPANCY_GRID_NUM_RING),
                                         ##"exists_pedestrian_1": spaces.MultiDiscrete([[0, 1]] * OCCUPANCY_GRID_NUM_RING),
                                         ##"exists_pedestrian_2": spaces.MultiDiscrete([[0, 1]] * OCCUPANCY_GRID_NUM_RING),
                                        #})
    traci.start(SUMO_CMD)
    pass
  def _step(self, action):
    try:
      if action["lane_change"] == 1 or anction["lane_change"] == 2:
        if traci.vehicle.couldChangeLane() == False:
          raise InvalidAction()
        if action["lane_change"] == 1:
          traci.vehicle.changeLane("ego", getLaneIndex+1, 1000)
        if action["lane_change"] == 2:
          traci.vehicle.changeLane("ego", getLaneIndex-1, 1000)
      traci.vehicle.slowdown("ego", action["speed_level"], 1000)
    traci.simulationStep()
    pass
  def _reset(self):
    traci.close()
    traci.start(SUMO_CMD)
    pass
  def _close(self):
    traci.colse()
