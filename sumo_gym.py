#!python3
__author__ = "Changjian Li"

SUMO_TOOLS_DIR = "/home/ken/project/sumo-0.30.0/tools"
SUMO_BIN = "/home/ken/project/sumo-0.30.0/bin/sumo"
SUMO_CONFIG = "/home/ken/project/sumo/test.sumocfg"
SUMO_CMD = [SUMO_BIN, "-c", SUMO_CONFIG, "--time-to-teleport", "-1"]
NET_XML_FILE = "/home/ken/project/sumo/test0.net.xml"

#OCCUPANCY_GRID_NUM_RING = 10
#OCCUPANCY_GRID_NUM_GRID_PER_RING = 8
#OCCUPANCY_GRID_NUM_GRID = OCCUPANCY_GRID_NUM_GRID_PER_RING * OCCUPANCY_GRID_NUM_RING

MAX_VEHICLE_SPEED = 200
MAX_LANE_PRIORITY = 2
NUM_VEHICLE_CONSIDERED = 16
OBSERVATION_RADIUS = 600

import gym
from gym import spaces
import numpy as np
import sys
import time
from math import pi
import re

try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import traci

from get_env_state import get_vehicle_state, Lanelet_graph, get_veh_next_edge_id

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
    3) "exists_vehicle": spaces.MultiDiscrete object, each of whose element indicates whether a vehicle exists[1] or not[0], we only
                         consider vehicles within the range 
    4) "speed": absolute speed of each vehicle
    5) "relative_position": space.Box object of size (NUM_VEHICLE_CONSIDERED, 2), each the position of the vehicle relative to the ego vehicle 
    6) "relative_heading": in radius, anticlockwise
    7) "has_priority"
    8) "dist_to_end_of_lane"
    9) "lane_relation": the relationship between the vehicle lane and the ego lane. Possible reltionships are:
         PEER[0]: share the same NEXT lane with the ego lane
         CONFLICT[1]: #CONFLICT if approaching the same intersection as the ego lane, and its route conflict that of the ego route
         CONFLICT_INTERSECTION[2]: the lane is already inside the intersection ego is approaching/is in, and conflicts with ego route
         NEXT[3]: the next lane of ego lane
         PREV[4]: the previous lane of ego lane
         LEFT[5]: to the left of ego lane
         RIGHT[6]: to the right of ego lane
         IRRELEVANT[7]: the lane is irrelevant to ego vehicle, e.g. opposite lane, in or approaching other intersections, 
                      or in the same intersection but with no conflict with ego route
         since a lane can have several of the above relationship with the ego lane at the same time, MultiBinary object of size 8 is needed per vehicle
  If we feed the observation space to a neural network, the weights connecting to the same key (of the observation space dictionary) should be identical
  If the value of "exists_vehicle" is zero for index i, then the value of all other keys at index i are set to zero
  The weight connection to key "exists_vehicle" should cancel out the effect of other keys, 
  so that if a vehicle doesn't exists at index i, then weighted sum for index i is zero 
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
                                         "exists_vehicle": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED)
                                         "speed": spaces.Box(0, MAX_VEHICLE_SPEED, (NUM_VEHICLE_CONSIDERED,))
                                         "relative_position": spaces.Box(float("-inf"), float("inf"), (NUM_VEHICLE_CONSIDERED, 2)),
                                         "relative_heading": spaces.Box(-pi, pi, (NUM_VEHICLE_CONSIDERED,)),
                                         "has_priority": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED)
                                         "dist_to_end_of_lane": spaces.Box(0, float("inf"), (NUM_VEHICLE_CONSIDERED,)),
                                         "lane_relation": spaces.MultiBinary(8 * NUM_VEHICLE_CONSIDERED)
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

def get_observation():
  veh_state_dict = get_vehicle_state()
  lanelet_graph = Lanelet_graph(NET_XML_FILE)
  obs_dict = {}
  obs_dict["ego_speed"] = veh_state_dict["ego"]["speed"]
  obs_dict["ego_dist_to_end_of_lane"] = veh_state_dict["ego"]["lane_length"] - veh_state_dict["ego"]["lane_position"]
  #lanes inside intersections have ids that start with ":"
  if veh_state_dict["ego"]["lane_id"][0] == ":":
    obs_dict["ego_in_intersection"] = 1
  else:
    obs_dict["ego_in_intersection"] = 0
  
  #vehicles inside the region of insterest (within the OBSERVATION_RADIUS) except for ego
  def in_ROI(ego_position, veh_position):
    if veh_position[0] > ego_position[0]-OBSERVATION_RADIUS and
       veh_position[1] > ego_position[1]-OBSERVATION_RADIUS and
       veh_position[0] < ego_position[0]+OBSERVATION_RADIUS and
       veh_position[1] < ego_position[1]+OBSERVATION_RADIUS:
      return True
    return False
  veh_state_dict_ROI = {k: v for k, v in veh_state_dict.items()
                        if k!="ego" and in_ROI(veh_state_dict["ego"]["position"], v["position"])
                        }
  
  #now deal with the relavant vehicles
  obs_dict["exists_vehicle"] = []
  obs_dict["speed"] = []
  obs_dict["relative_position"] = []
  obs_dict["relative_heading"] = []
  obs_dict["has_priority"] = []
  obs_dict["dist_to_end_of_lane"] = []
  obs_dict["lane_relation"] = []

  ego_next_edge_id = get_next_edge_id("ego")
  lane_id_list_ego_next_edge = lanelet_graph.get_lane_id_list_in_edge(ego_next_edge_id)
  lane_id_list_ego_edge = lanelet_graph.get_lane_id_list_in_edge(veh_state_dict["ego"]["edge_id"])
  for veh_id, state_dict in veh_state_dict_ROI.items():
    obs_dict["exists_vehicle"] += [1]
    obs_dict["speed"] += [state_dict["speed"]]
    obs_dict["relative_position"] += [state_dict["position"] - veh_state_dict["ego"]["position"]]
    obs_dict["relative_heading"] += [-(state_dict["angle"] - veh_state_dict["ego"]["angle"])/180 * pi]
    if lanelet_graph.get_priority(state_dict["lane_id"]) > lanelet_graph.get_priority(veh_state_dict["ego"]["lane_id"]):
      obs_dict["has_priority"] += [1]
    else:
      obs_dict["has_priority"] += [1]
    obs_dict["dist_to_end_of_lane"] += state_dict["lane_length"] - state_dict["lane_position"]
    #check if the each of the possible relationship holds for the vehicle lane 
    relation_list = [0] * 8
    lane_id_list_veh_edge = lanelet_graph.get_lane_id_list_in_edge(state_dict["edge_id"])
    veh_next_edge_id = get_next_edge_id(veh_id)
    lane_id_list_veh_next_edge = lanelet_graph.get_lane_id_list_in_edge(veh_next_edge_id)
    #PEER if vehicle share the same next lane, since we only have edge (not lane) information within the route, we need to
    #search inside the next edge to see if there're any lanes whose previous lane belongs to the current edge of ego and veh
    if veh_next_edge_id == ego_next_edge_id and ego_next_edge_id != None:
      for x in lane_id_list_ego_next_edge:
        prev_lane_id_set = set(lanelet_graph.get_prev_lane_id_list(x))
        if len(prev_lane_id_set & set(lane_id_list_ego_edge)) > 0 and
           len(prev_lane_id_set & set(lane_id_list_veh_edge)) > 0:
             relation_list[0] = 1 #PEER
             break
    
    #if the to lanes are appoaching the same intersection
    def intersect(P0, P1, Q0, Q1):
      def ccw(A,B,C):
        """check if the three points are in counterclockwise order"""
        return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
      return ccw(P0,Q0,Q1) != ccw(P1,Q0,Q1) and ccw(P0,P1,Q0) != ccw(P0,P1,Q1)
    if lanelet_graph.get_to_node_id(veh_state_dict["ego"]["lane_id"]) == lanelet_graph.get_to_node_id(state_dict["lane_id"]):
      #CONFLICT if approaching the same intersection as the ego lane, and its route conflict that of the ego route
      for u in lane_id_list_veh_next_edge:
        for p in lanelet_graph.get_prev_lane_id_list(u):
          for v in lane_id_list_ego_next_edge:
            for q in lanelet_graph.get_prev_lane_id_list(v):
              if intersect(lanelet_graph.get_waypoint(u), lanelet_graph.get_waypoint(p), 
                           lanelet_graph.get_waypoint(v), lanelet_graph.get_waypoint(q)
                           ):
                if state_dict["lane_id"][0] != ":":
                  relation_list[1] = 1 #CONFLICT
                else:
                  relation_list[2] = 1 #CONFLICT_INTERSECTION
    
    #NEXT, PREV, LEFT, RIGHT
    if state_dict["lane_id"] in lane_id_list_ego_next_edge:
      if len(set(lanelet_graph.get_prev_lane_id_list(state_dict["lane_id"])) & set(lane_id_list_ego_edge)) > 0:
        relation_list[3] = 1 #NEXT
    if veh_next_edge_id == veh_state_dict["ego"]["edge_id"]:
      if len(set(lanelet_graph.get_prev_lane_id_list(veh_state_dict["ego"]["lane_id"])) & set(lane_id_list_veh_edge)) > 0:
        relation_list[4] = 1 #PREV
    if state_dict["lane_id"] == lanelet_graph.get_left_lane_id:
      relation_list[5] = 1 #LEFT
    if state_dict["lane_id"] == lanelet_graph.get_right_lane_id:
      relation_list[6] = 1 #RIGHT
    if sum(relation_list[:-1]) == 0:
      relation_list[7] = 1 #IRRELEVANT
    
    obs_dict["lane_relation"] += relation_list
  pass
