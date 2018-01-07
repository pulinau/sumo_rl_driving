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

observation_space = spaces.Dict({"ego_speed": spaces.Box(0, MAX_VEH_SPEED, shape=(1,)),
             "ego_dist_to_end_of_lane": spaces.Box(0, float("inf"), shape=(1,)),
             "ego_in_intersection": spaces.Discrete(2),
             "ego_exists_left_lane": spaces.Discrete(2),
             "ego_exists_right_lane": spaces.Discrete(2),
             "ego_correct_lane_gap": spaces.Box(float("-inf"), float("inf"), shape=(1,)),
             "exists_vehicle": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "speed": spaces.Box(0, MAX_VEH_SPEED, (NUM_VEHICLE_CONSIDERED,)),  # absolute speed
             "dist_to_end_of_lane": spaces.Box(0, float("inf"), (NUM_VEHICLE_CONSIDERED,)),
             "in_intersection": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "relative_position": spaces.Box(-OBSERVATION_RADIUS, OBSERVATION_RADIUS, (NUM_VEHICLE_CONSIDERED, 2)),
             "relative_heading": spaces.Box(-pi, pi, (NUM_VEHICLE_CONSIDERED,)),
             "has_priority": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "veh_relation": spaces.MultiBinary(NUM_VEH_RELATION * NUM_VEHICLE_CONSIDERED)
             })

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
    veh_dict[veh_id]["lane_index"] = traci.vehicle.getLaneIndex(veh_id)
    veh_dict[veh_id]["lane_length"] = traci.lane.getLength(veh_dict[veh_id]["lane_id"]) 
    veh_dict[veh_id]["lane_position"] = traci.vehicle.getLanePosition(veh_id) # position in the lane
    veh_dict[veh_id]["route"] = traci.vehicle.getRoute(veh_id)
    
    route = veh_dict[veh_id]["route"] # route is an edge id list of the vehicle's route
    if len(route) > traci.vehicle.getRouteIndex(veh_id) + 1: 
      veh_dict[veh_id]["next_normal_edge_id"] = route[traci.vehicle.getRouteIndex(veh_id) + 1]
    else:
      veh_dict[veh_id]["next_normal_edge_id"] = None
  
  return veh_dict

def get_lanelet_dict(sumo_net_xml_file):
  net = sumolib.net.readNet(sumo_net_xml_file, withInternal=True)
  lanelet_dict = {}
  edges = net.getEdges()
  for edge in edges:
    # add "next" and "previous" connection
    # since there's no direct sumolib API to get previous lane, we need to do some tricks here
    for lane in edge.getLanes():
      lane_id = lane.getID()
      lanelet_dict[lane_id] = {}
      lanelet_dict[lane_id]["prev_lane_id_list"] = []
      lanelet_dict[lane_id]["prev_normal_lane_id_list"] = []
  for edge in edges:
    for lane in edge.getLanes():
      lane_id = lane.getID()
      lane_index = lane.getIndex()
      lanelet_dict[lane_id]["waypoint"] = lane.getShape()
      lanelet_dict[lane_id]["from_node_id"] = edge.getFromNode().getID()
      lanelet_dict[lane_id]["to_node_id"] = edge.getToNode().getID()
      lanelet_dict[lane_id]["edge_id"] = edge.getID()
      lanelet_dict[lane_id]["lane_index"] = lane_index
      
      if lane_id[0] == ':':
        lanelet_dict[lane_id]["edge_priority"] = float("inf")
      else:
        lanelet_dict[lane_id]["edge_priority"] = edge.getPriority()        
      
      lanelet_dict[lane_id]["next_normal_lane_id_list"] = [conn.getToLane().getID() for conn in lane.getOutgoing()]
      if lane_id[0] == ':':
        lanelet_dict[lane_id]["next_lane_id_list"] = [conn.getToLane().getID() for conn in lane.getOutgoing()]
      else:
        lanelet_dict[lane_id]["next_lane_id_list"] = [conn.getViaLaneID() for conn in lane.getOutgoing()] 
        for next_lane_id in lanelet_dict[lane_id]["next_normal_lane_id_list"] + lanelet_dict[lane_id]["next_lane_id_list"]:
           lanelet_dict[next_lane_id]["prev_normal_lane_id_list"] += [lane_id]        
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
  
  # now ignore the internal edges/lanes
  net = sumolib.net.readNet(sumo_net_xml_file, withInternal=False)
  edges = net.getEdges()
  for edge in edges:
    for lane in edge.getLanes():
      lane_id = lane.getID()
      lanelet_dict[lane_id]["next_normal_lane_id_list"] = [conn.getToLane().getID() for conn in lane.getOutgoing()] 
      for next_lane_id in lanelet_dict[lane_id]["next_lane_id_list"]:
        lanelet_dict[next_lane_id]["prev_normal_lane_id_list"] += [lane_id]
  
  return lanelet_dict
  
def get_edge_dict(sumo_net_xml_file):
  net = sumolib.net.readNet(sumo_net_xml_file, withInternal=True)
  edge_dict = {}
  edges = net.getEdges()
  for edge in edges:
    edge_id = edge.getID()
    edge_dict[edge_id] = {}
    edge_dict[edge_id]["lane_id_list"] = [lane.getID() for lane in edge.getLanes()]
    edge_dict[edge_id]["from_node_id"] = edge.getFromNode().getID()
    edge_dict[edge_id]["to_node_id"] = edge.getToNode().getID()
    # intersection has the highest priority
    if edge_id[0] == ':':
      edge_dict[edge_id]["priority"] = float("inf")
    else:
      edge_dict[edge_id]["priority"] = edge.getPriority()
    
  return edge_dict

def get_obs():
  
  veh_dict = get_veh_dict()
  lanelet_dict = get_lanelet_dict(NET_XML_FILE)
  edge_dict = get_edge_dict(NET_XML_FILE)
  
  obs_dict = {}
  
  ego_dict = veh_dict[EGO_VEH_ID]
  obs_dict["ego_speed"] = ego_dict["speed"]
  obs_dict["ego_dist_to_end_of_lane"] = ego_dict["lane_length"] - ego_dict["lane_position"]
  
  # lanes inside intersections have ids that start with ":"
  if ego_dict["lane_id"][0] == ":":
    obs_dict["ego_in_intersection"] = 1
  else:
    obs_dict["ego_in_intersection"] = 0

  if traci.vehicle.couldChangeLane(EGO_VEH_ID, 1):
    obs_dict["ego_exists_left_lane"] = 1
  else:
    obs_dict["ego_exists_left_lane"] = 0
  if traci.vehicle.couldChangeLane(EGO_VEH_ID, -1):
    obs_dict["ego_exists_right_lane"] = 1
  else:
    obs_dict["ego_exists_right_lane"] = 0
  
  lane_id_list_ego_edge = edge_dict[ego_dict["edge_id"]]["lane_id_list"]
  if ego_dict["next_normal_edge_id"] != None:
    lane_id_list_ego_next_normal_edge = edge_dict[ego_dict["next_normal_edge_id"]]["lane_id_list"]
  else:
    lane_id_list_ego_next_normal_edge = []
  # correct lane
  # if next normal edge doesn't exist, consider ego to be already in correct lane
  obs_dict["ego_correct_lane_gap"] = 0
  for x in lane_id_list_ego_next_normal_edge:
    for y in lane_id_list_ego_edge:
      if internal_lane_id_between_lanes(y, x, lanelet_dict)!= None:
        obs_dict["ego_correct_lane_gap"] = lanelet_dict[y]["lane_index"] - ego_dict["lane_index"]
  
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
                        if k!=EGO_VEH_ID and in_ROI(ego_dict["position"], v["position"])
                        }

  # now deal with the relavant vehicles
  obs_dict["exists_vehicle"] = [0] * NUM_VEHICLE_CONSIDERED
  obs_dict["speed"] = [0] * NUM_VEHICLE_CONSIDERED
  obs_dict["dist_to_end_of_lane"] = [0] * NUM_VEHICLE_CONSIDERED
  obs_dict["in_intersection"] = [0] * NUM_VEHICLE_CONSIDERED
  obs_dict["relative_position"] = [[0, 0]] * NUM_VEHICLE_CONSIDERED
  obs_dict["relative_heading"] = [0] * NUM_VEHICLE_CONSIDERED
  obs_dict["has_priority"] = [0] * NUM_VEHICLE_CONSIDERED
  obs_dict["veh_relation"] = [0] * (NUM_VEH_RELATION * NUM_VEHICLE_CONSIDERED)

  
  for veh_index, (veh_id, state_dict) in enumerate(veh_dict_ROI.items()):
    if veh_index >= NUM_VEHICLE_CONSIDERED:
      break
    
    obs_dict["exists_vehicle"][veh_index] = 1
    obs_dict["speed"][veh_index] = state_dict["speed"]
    obs_dict["dist_to_end_of_lane"][veh_index] = state_dict["lane_length"] - state_dict["lane_position"]
    
    if state_dict["edge_id"][0] == ':':
      obs_dict["in_intersection"][veh_index] = 1
    
    # transform the position to ego coordinate
    ego_angle_rad = ego_dict["angle"]/180 * pi
    rotation_mat = np.array([[np.cos(ego_angle_rad), -np.sin(ego_angle_rad)],
                             [np.sin(ego_angle_rad), np.cos(ego_angle_rad)]])
    relative_position = np.array(state_dict["position"]) - np.array(ego_dict["position"])
    relative_position = np.matmul(rotation_mat,relative_position)
    obs_dict["relative_position"][veh_index] = relative_position
    
    relative_heading = -(state_dict["angle"] - ego_dict["angle"])/180 * pi
    if relative_heading > pi:
      relative_heading -= 2*pi
    elif relative_heading < -pi:
      relative_heading += 2*pi
    obs_dict["relative_heading"][veh_index] = relative_heading
    
    # vehicle has priority over ego if the vehicle is
    # approaching/in the same intersection and it's inside a lane of higher priority
    if edge_dict[state_dict["edge_id"]]["to_node_id"] == edge_dict[ego_dict["edge_id"]]["to_node_id"] and \
       edge_dict[state_dict["edge_id"]]["priority"] > edge_dict[ego_dict["edge_id"]]["priority"]:
      obs_dict["has_priority"][veh_index] = 1
    
    # check if the each of the possible relationship holds for the vehicle
    relation_list = [0] * NUM_VEH_RELATION
    lane_id_list_veh_edge = edge_dict[state_dict["edge_id"]]["lane_id_list"]
    if state_dict["next_normal_edge_id"] != None:
      lane_id_list_veh_next_normal_edge = edge_dict[state_dict["next_normal_edge_id"]]["lane_id_list"]
    else:
      lane_id_list_veh_next_normal_edge = []
    # PEER if vehicle share the same next lane, since we only have edge (not lane) information within the route, we need to
    # search inside the next edge to see if there're any lanes whose previous lane belongs to the current edge of veh
    if state_dict["next_normal_edge_id"] == ego_dict["next_normal_edge_id"] and ego_dict["next_normal_edge_id"] != None:
      for x in lane_id_list_ego_next_normal_edge:
        for y in lane_id_list_veh_edge:
          if internal_lane_id_between_lanes(y, x, lanelet_dict) != None:
            relation_list[0] = 1 # PEER
    
    # CONFLICT if approaching/in the same intersection as the ego lane, and its route conflict that of the ego route
    if edge_dict[ego_dict["edge_id"]]["to_node_id"] == edge_dict[state_dict["edge_id"]]["to_node_id"]:
      for u in lane_id_list_veh_next_normal_edge:
        for v in lane_id_list_veh_edge:
          lane_id0 = internal_lane_id_between_lanes(v, u, lanelet_dict)
          for p in lane_id_list_ego_next_normal_edge:
            for q in lane_id_list_ego_edge:
              lane_id1 = internal_lane_id_between_lanes(q, p, lanelet_dict)
              if lane_id0 != None and lane_id1 != None:
                if waypoint_intersect(lanelet_dict[lane_id0]["waypoint"], lanelet_dict[lane_id1]["waypoint"]) == True:
                  relation_list[1] = 1 # CONFLICT
    
    # NEXT, LEFT, RIGHT
    if state_dict["lane_id"] in lanelet_dict[ego_dict["lane_id"]]["next_lane_id_list"]:
      relation_list[2] = 1 # NEXT
    if state_dict["lane_id"] == lanelet_dict[ego_dict["lane_id"]]["left_lane_id"]:
      relation_list[3] = 1 # LEFT
    if state_dict["lane_id"] == lanelet_dict[ego_dict["lane_id"]]["right_lane_id"]:
      relation_list[4] = 1 # RIGHT
    
    if state_dict["lane_id"] == ego_dict["lane_id"]:
      if state_dict["lane_position"] > ego_dict["lane_id"]:
        relation_list[5] = 1 # AHEAD
      else:
        relation_list[6] = 1 # BEHIND
    
    if sum(relation_list[:-1]) == 0:
      relation_list[-1] = 1 # IRRELEVANT
    
    obs_dict["veh_relation"][(veh_index * NUM_VEH_RELATION):((veh_index+1) * NUM_VEH_RELATION)] = relation_list
  
  pass
  return obs_dict

def intersect(p0, p1, q0, q1):
  """check if two line segments p0-p1, q0-q1 intersect"""
  def ccw(a,b,c):
    """check if the three points are in counterclockwise order"""
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
  return ccw(p0,q0,q1) != ccw(p1,q0,q1) and ccw(p0,p1,q0) != ccw(p0,p1,q1)

def waypoint_intersect(waypoints0, waypoints1):
  for m in range(len(waypoints0)-1):
    for n in range(len(waypoints1)-1):
      if intersect(waypoints0[m], waypoints0[m+1], waypoints1[n], waypoints1[n+1]):
        return True
  return  False

def internal_lane_id_between_lanes(from_lane_id, to_lane_id, lanelet_dict):
  if from_lane_id[0] == ':':
    if to_lane_id in lanelet_dict[from_lane_id]["next_lane_id_list"]:
      return from_lane_id
  elif to_lane_id[0] == ':':
    if to_lane_id in lanelet_dict[from_lane_id]["next_lane_id_list"]:
      return to_lane_id
  else:
    for lane_id in lanelet_dict[from_lane_id]["next_lane_id_list"]:
      if lane_id in lanelet_dict[to_lane_id]["prev_lane_id_list"]:
        return lane_id
    return None
  
