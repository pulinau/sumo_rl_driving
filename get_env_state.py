#!python3
__author__ = "Changjian Li"

SUMO_TOOLS_DIR = "/home/ken/project/sumo-0.30.0/tools"
EGO_VEH_ID = "ego"

import sys
try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")

import traci
import sumolib


def get_vehicle_state():
  """
  get the current state of all the vehicles.
  Returns:
    veh_state_dict:
    {"type": "vehicle", "position": ...}
  """
  veh_id_list = traci.vehicle.getIDList()
  veh_state_dict = {}
  for veh_id in veh_id_list:
    veh_state_dict[veh_id] = {}
    veh_state_dict[veh_id]["type"] = "vehicle"
    veh_state_dict[veh_id]["position"] = traci.vehicle.getPosition(veh_id)
    veh_state_dict[veh_id]["angle"] = traci.vehicle.getAngle(veh_id) #in degree. North is zero, clockwise
    veh_state_dict[veh_id]["speed"] = traci.vehicle.getSpeed(veh_id)
    veh_state_dict[veh_id]["dimension"]  = (traci.vehicle.getLength(veh_id),traci.vehicle.getWidth(veh_id))
    veh_state_dict[veh_id]["edge_id"] = traci.vehicle.getRoadID(veh_id)
    veh_state_dict[veh_id]["lane_id"] = traci.vehicle.getLaneID(veh_id)
    veh_state_dict[veh_id]["lane_length"] = traci.lane.getLength(veh_state_dict[veh_id]["lane"]) #position in the lane
    veh_state_dict[veh_id]["lane_position"] = traci.vehicle.getLanePosition(veh_id)
    veh_state_dict[veh_id]["route"] = traci.route.getEdges(traci.vehicle.getRouteID(veh_id))
  return veh_state_dict

class lanelet_graph:
  """
  lanelet graph class 
  """
  def __init__(self):
    """
    Create the lanelet graph from net.xml file using sumolib.
    the lanelet graph is represented as a dictionary of the following structure:
      self.lanelet_dict:
        {"lanelet_id0": {"waypoint": [p0, p1...], 
                        "from_node_id": from_node_id, 
                        "to_node_id": to_node_id, 
                        "edge_id": edge_id,
                        "next_lane_id_list": next_lane_id_list,
                        "previous_lane_id_list": previous_lane_id_list,
                        "left_lane_id": left_lane_id,
                        "right_lane_id": right_lane_id,
                         }, ...
         }
    Args:
      file path of net.xml file
    """
    net = sumolib.net.readNet(sumo_net_xml_file)
    self.lanelet_dict = {}
    edges = net.getEdges()

    for edge in edges:
      # add "next" and "previous" connection
      # since there's no direction sumolib API to get previous lane, we need to do some tricks here
      for lane in edge.getLanes():
        lane_id = lane.getID()
        self.lanelet_dict[lane_id] = {}
        self.lanelet_dict[lane_id]["previous_lane_id_list"] = []
      for lane_index, lane in enumerate(edge.getLanes()):
        lane_id = lane.getID()
        self.lanelet_dict[lane_id]["waypoint"] = traci.lane.getShape(lane_id)
        self.lanelet_dict[lane_id]["from_node_id"] = edge.getFromNode().getID()
        self.lanelet_dict[lane_id]["to_node_id"] = edge.getToNode().getID()
        self.lanelet_dict[lane_id]["edge_id"] = edge.getID()
        self.lanelet_dict[lane_id]["next_lane_id_list"] = [conn.getToLane().getID() for conn in lane.getOutgoing()] 
        for next_lane_id in self.lanelet_dict[lane_id]["next_lane_id_list"]:
          self.lanelet_dict[next_lane_id]["previous_lane_id_list"] += [lane_id]
        if lane_index == len(edge.getLanes()):
          self.lanelet_dict[lane_id]["left_lane_id"] = None
        else:
          self.lanelet_dict[lane_id]["left_lane_id"] = edge.getLanes()[lane_index+1].getID()
        if lane_index == 0:
          self.lanelet_dict[lane_id]["right_lane_id"] = None
        else
          self.lanelet_dict[lane_id]["right_lane_id"] = edge.getLanes()[lane_index-1].getID()
      #"left" and "right" connection for opposite direction lane is not added
    pass
  
  def get_next_lane_id_list(self, lane_id):
    return self.lanelet_dict[lane_id]["next_lane_id_list"]
  
  def get_previous_lane_id_list(self, lane_id):
    return self.lanelet_dict[lane_id]["previous_lane_id_list"]
  
  def get_left_lane_id(self, lane_id):
    return self.lanelet_dict[lane_id]["left_lane_id"]

  def get_right_lane_id(self, lane_id):
    return self.lanelet_dict[lane_id]["right_lane_id"]
  
  def get_waypoint(self, lane_id):
    return self.lanelet_dict[lane_id]["waypoint"]
  
  def get_from_node_id(self, lane_id):
    return self.lanelet_dict[lane_id]["from_node_id"]
  
  def get_to_node_id(self, lane_id):
    return self.lanelet_dict[lane_id]["to_node_id"]
  
  def get_edge_id(self, lane_id):
    return self.lanelet_dict[lane_id]["edge_id"]
  
  def get_lane_id_list_in_edge(self, edge_id):
    return [lane_id for lane_id in lanelet_dict if lanelet_dict[lane_id]["edge_id"] == edge_id]
