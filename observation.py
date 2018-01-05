#!python3
__author__ = "Changjian Li"

import numpy as np
import sys
import time
from math import pi

import gym
from gym import spaces

from parameters import *
try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import traci

import sumolib

def get_veh_dict():
  """get the current state of all vehicles."""
  veh_id_list = traci.vehicle.getIDList()
  veh_dict = {}
  
  for veh_id in veh_id_list:
    veh_dict[veh_id] = {}
    veh_dict[veh_id]["type"] = "vehicle"
    veh_dict[veh_id]["position"] = traci.vehicle.getPosition(veh_id)
    veh_dict[veh_id]["angle"] = traci.vehicle.getAngle(veh_id) # in degree. North is zero, clockwise
    veh_dict[veh_id]["speed"] = traci.vehicle.getSpeed(veh_id)
    veh_dict[veh_id]["dimension"]  = (traci.vehicle.getLength(veh_id),traci.vehicle.getWidth(veh_id))
    veh_dict[veh_id]["edge_id"] = traci.vehicle.getRoadID(veh_id)
    veh_dict[veh_id]["lane_id"] = traci.vehicle.getLaneID(veh_id)
    veh_dict[veh_id]["lane_length"] = traci.lane.getLength(veh_dict[veh_id]["lane_id"]) 
    veh_dict[veh_id]["lane_position"] = traci.vehicle.getLanePosition(veh_id) # position in the lane
    veh_dict[veh_id]["route"] = traci.vehicle.getRoute(veh_id)
    
    route = veh_dict[veh_id]["route"] # route is an edge id list of the vehicle's route
    if len(route) > traci.vehicle.getRouteIndex(veh_id) + 1: 
      veh_dict[veh_id]["next_edge_id"] = route[traci.vehicle.getRouteIndex(veh_id) + 1]
    else:
      veh_dict[veh_id]["next_edge_id"] = None
  
  return veh_dict

def get_lanelet_dict(sumo_net_xml_file):
  net = sumolib.net.readNet(sumo_net_xml_file)
  lanelet_dict = {}
  edges = net.getEdges()
  for edge in edges:
    # add "next" and "previous" connection
    # since there's no direct sumolib API to get previous lane, we need to do some tricks here
    for lane in edge.getLanes():
      lane_id = lane.getID()
      lanelet_dict[lane_id] = {}
      lanelet_dict[lane_id]["prev_lane_id_list"] = []
  for edge in edges:
    for lane in edge.getLanes():
      lane_id = lane.getID()
      lane_index = lane.getIndex()
      lanelet_dict[lane_id]["waypoint"] = traci.lane.getShape(lane_id)
      lanelet_dict[lane_id]["from_node_id"] = edge.getFromNode().getID()
      lanelet_dict[lane_id]["to_node_id"] = edge.getToNode().getID()
      lanelet_dict[lane_id]["edge_id"] = edge.getID()
      lanelet_dict[lane_id]["edge_priority"] = edge.getPriority()
      lanelet_dict[lane_id]["next_lane_id_list"] = [conn.getToLane().getID() for conn in lane.getOutgoing()] 
      for next_lane_id in lanelet_dict[lane_id]["next_lane_id_list"]:
        lanelet_dict[next_lane_id]["prev_lane_id_list"] += [lane_id]
      if lane_index == len(edge.getLanes()) - 1:
        lanelet_dict[lane_id]["left_lane_id"] = None
      else:
        lanelet_dict[lane_id]["left_lane_id"] = edge.getLanes()[lane_index+1].getID()
      if lane_index == 0:
        lanelet_dict[lane_id]["right_lane_id"] = None
      else:
        lanelet_dict[lane_id]["right_lane_id"] = edge.getLanes()[lane_index-1].getID()
    # "left" and "right" connections for opposite direction lane are not added
  return lanelet_dict
  
def get_edge_dict(sumo_net_xml_file):
  net = sumolib.net.readNet(sumo_net_xml_file)
  edge_dict = {}
  edges = net.getEdges()
  for edge in edges:
    edge_id = edge.getID()
    edge_dict[edge_id] = {}
    edge_dict[edge_id]["lane_id_list"] = [lane.getID() for lane in edge.getLanes()]
  return edge_dict

observation_space = spaces.Dict({"ego_speed": spaces.Box(0, MAX_VEH_SPEED, shape=(1,)),
             "ego_dist_to_end_of_lane": spaces.Box(0, float("inf"), shape=(1,)),
             "ego_in_intersection": spaces.Discrete(2),
             "ego_exists_left_lane": spaces.Discrete(2),
             "ego_exists_right_lane": spaces.Discrete(2),
             "ego_correct_lane": spaces.Discrete(2 * NUM_LANE_CONSIDERED + 1),
             "exists_vehicle": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "speed": spaces.Box(0, MAX_VEH_SPEED, (NUM_VEHICLE_CONSIDERED,)),  # absolute speed
             "relative_position": spaces.Box(-OBSERVATION_RADIUS, OBSERVATION_RADIUS, (NUM_VEHICLE_CONSIDERED, 2)),
             "relative_heading": spaces.Box(-pi, pi, (NUM_VEHICLE_CONSIDERED,)),
             "has_priority": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "dist_to_end_of_lane": spaces.Box(0, float("inf"), (NUM_VEHICLE_CONSIDERED,)),
             "veh_relation": spaces.MultiBinary(NUM_VEH_RELATION * NUM_VEHICLE_CONSIDERED)
             })

def get_obs():
  if "self.old_veh_dict" not in locals():
    # old copy of veh_dict that retains the edge_id the vehicle was in before entering an intersection
    self.old_veh_dict = {} 
  veh_dict = get_veh_dict()
  for veh_id, state_dict in veh_dict.items():
    if state_dict["edge_id"][0] != ':':
      self.old_veh_dict[veh_id] = state_dict
  
  lanelet_dict = get_lanelet_dict(NET_XML_FILE)
  edge_dict = get_edge_dict(NET_XML_FILE)
  
  obs_dict = {}
  
  obs_dict["ego_speed"] = veh_dict[EGO_VEH_ID]["speed"]
  obs_dict["ego_dist_to_end_of_lane"] = veh_dict[EGO_VEH_ID]["lane_length"] - veh_dict[EGO_VEH_ID]["lane_position"]
  
  # lanes inside intersections have ids that start with ":"
  if veh_dict[EGO_VEH_ID]["lane_id"][0] == ":":
    obs_dict["ego_in_intersection"] = 1
  else:
    obs_dict["ego_in_intersection"] = 0
  
  if traci.vehicle.couldChangeLane(EGO_VEH_ID, 1) == True:
    obs_dict["ego_exists_left_lane"] = 1
  else:
    obs_dict["ego_exists_left_lane"] = 0
  if traci.vehicle.couldChangeLane(EGO_VEH_ID, 0) == True:
    obs_dict["ego_exists_right_lane"] = 1
  else:
    obs_dict["ego_exists_right_lane"] = 0
  
  # correct lane
  
  # vehicles inside region of insterest
  def in_ROI(ego_position, veh_position):
    if ((veh_position[0] > ego_position[0]-OBSERVATION_RADIUS) and 
        (veh_position[1] > ego_position[1]-OBSERVATION_RADIUS) and 
        (veh_position[0] < ego_position[0]+OBSERVATION_RADIUS) and 
        (veh_position[1] < ego_position[1]+OBSERVATION_RADIUS)
        ):
      return True
    return False
  veh_dict_ROI = {k: v for k, v in veh_dict.items()
                        if k!=EGO_VEH_ID and in_ROI(veh_dict[EGO_VEH_ID]["position"], v["position"])
                        }
  
  # now deal with the relavant vehicles
  obs_dict["exists_vehicle"] = []
  obs_dict["speed"] = []
  obs_dict["relative_position"] = []
  obs_dict["relative_heading"] = []
  obs_dict["has_priority"] = []
  obs_dict["dist_to_end_of_lane"] = []
  obs_dict["veh_relation"] = []

  ego_next_edge_id = veh_dict[EGO_VEH_ID]["next_edge_id"]
  lane_id_list_ego_next_edge = edge_dict[ego_next_edge_id]["lane_id_list"]
  lane_id_list_ego_edge = edge_dict[self.old_veh_dict[EGO_VEH_ID]["edge_id"]]["lane_id_list"]
  for veh_id, state_dict in veh_dict_ROI.items():
    obs_dict["exists_vehicle"] += [1]
    obs_dict["speed"] += [state_dict["speed"]]
    obs_dict["relative_position"] += [np.array(state_dict["position"]) - np.array(veh_dict[EGO_VEH_ID]["position"])]
    relative_heading = -(state_dict["angle"] - veh_dict[EGO_VEH_ID]["angle"])/180 * pi
    if relative_heading > pi:
      relative_heading -= 2*pi
    elif relative_heading < -pi:
      relative_heading += 2*pi
    obs_dict["relative_heading"] += [relative_heading]
    
    # vehicle has priority over ego if ego is not in intersection, the vehicle is approaching/in the same intersection and:
    # it's inside the intersection while ego is not, or
    # it's inside a lane of higher priority than ego
    if ((obs_dict["ego_in_intersection"] == 0) and 
        (lanelet_dict[self.old_veh_dict[EGO_VEH_ID]["lane_id"]]["to_node_id"] == lanelet_dict[self.old_veh_dict[veh_id]["lane_id"]]["to_node_id"])
        ):
      if ((veh_dict[EGO_VEH_ID["edge_id"][0]] == ':') or
          (lanelet_dict[self.old_veh_dict[veh_id]["lane_id"]]["edge_priority"] > lanelet_dict[self.old_veh_dict[EGO_VEH_ID]["lane_id"]]["edge_priority"])
          ):
        obs_dict["has_priority"] += [1]
    else:
      obs_dict["has_priority"] += [0]
    
    obs_dict["dist_to_end_of_lane"] += [state_dict["lane_length"] - state_dict["lane_position"]]
    
    # check if the each of the possible relationship holds for the vehicle lane
    relation_list = [0] * NUM_VEH_RELATION
    lane_id_list_veh_edge = edge_dict[self.old_veh_dict[veh_id]["edge_id"]]["lane_id_list"]
    veh_next_edge_id = state_dict["next_edge_id"]
    lane_id_list_veh_next_edge = edge_dict[veh_next_edge_id]["lane_id_list"]
    #PEER if vehicle share the same next lane, since we only have edge (not lane) information within the route, we need to
    #search inside the next edge to see if there're any lanes whose previous lane belongs to the current edge of ego and veh
    if veh_next_edge_id == ego_next_edge_id and ego_next_edge_id != None:
      for x in lane_id_list_ego_next_edge:
        prev_lane_id_set = set(lanelet_dict[x]["prev_lane_id_list"])
        if ((len(prev_lane_id_set & set(lane_id_list_ego_edge)) > 0) and
            (len(prev_lane_id_set & set(lane_id_list_veh_edge)) > 0)
            ):
             relation_list[0] = 1 #PEER
             break
    
    # if the to lanes are appoaching the same intersection
    def intersect(P0, P1, Q0, Q1):
      def ccw(A,B,C):
        """check if the three points are in counterclockwise order"""
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
      return ccw(P0,Q0,Q1) != ccw(P1,Q0,Q1) and ccw(P0,P1,Q0) != ccw(P0,P1,Q1)
    
    if lanelet_dict[self.old_veh_dict[EGO_VEH_ID]["lane_id"]]["to_node_id"] == lanelet_dict[self.old_veh_dict[veh_id]["lane_id"]]["to_node_id"]:
      # CONFLICT if approaching the same intersection as the ego lane, and its route conflict that of the ego route
      for u in lane_id_list_veh_next_edge:
        for p in lanelet_dict[u]["prev_lane_id_list"]:
          for v in lane_id_list_ego_next_edge:
            for q in lanelet_dict[v]["prev_lane_id_list"]:
              if intersect(lanelet_dict[u]["waypoint"][0], lanelet_dict[p]["waypoint"][1], 
                           lanelet_dict[v]["waypoint"][0], lanelet_dict[q]["waypoint"][1]
                           ):
                if state_dict["lane_id"][0] != ":":
                  relation_list[1] = 1 # CONFLICT
                else:
                  relation_list[2] = 1 # CONFLICT_INTERSECTION
    
    # NEXT, PREV, LEFT, RIGHT
    if self.old_veh_dict[veh_id]["lane_id"] in lane_id_list_ego_next_edge:
      if len(set(lanelet_dict[self.old_veh_dict[veh_id]["lane_id"]]["prev_lane_id_list"]) & set(lane_id_list_ego_edge)) > 0:
        relation_list[3] = 1 # NEXT
    if veh_next_edge_id == self.old_veh_dict[EGO_VEH_ID]["edge_id"]:
      if len(set(lanelet_dict[self.old_veh_dict[EGO_VEH_ID]["lane_id"]["prev_lane_id_list"]) & set(lane_id_list_veh_edge)) > 0:
        relation_list[4] = 1 # PREV
    
    if veh_dict[EGO_VEH_ID]["lane_id"][0] != ':':
      if state_dict["lane_id"] == lanelet_dict[veh_dict[EGO_VEH_ID]["lane_id"]]["left_lane_id"]:
        relation_list[5] = 1 # LEFT
      if state_dict["lane_id"] == lanelet_dict[veh_dict[EGO_VEH_ID]["lane_id"]]["right_lane_id"]:
        relation_list[6] = 1 # RIGHT
    
    if state_dict["lane_id"] == veh_dict[EGO_VEH_ID]["lane_id"]:
      if state_dict["lane_position"] > veh_dict[EGO_VEH_ID]["lane_id"]:
        relation_list[7] = 1 # AHEAD
      else:
        relation_list[8] = 1 # BEHIND
    if sum(relation_list[:-1]) == 0:
      relation_list[9] = 1 # IRRELEVANT
    
    obs_dict["veh_relation"] += relation_list
  
  pass
  return obs_dict
