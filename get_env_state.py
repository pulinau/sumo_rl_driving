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
    veh_state_dict[veh_id]["edge"] = traci.vehicle.getRoadID(veh_id)
    veh_state_dict[veh_id]["lane"] = traci.vehicle.getLaneID(veh_id)
    veh_state_dict[veh_id]["lane_length"] = traci.lane.getLength(veh_state_dict[veh_id]["lane"]) #position in the lane
    veh_state_dict[veh_id]["lane_position"] = traci.vehicle.getLanePosition(veh_id)
    veh_state_dict[veh_id]["route"] = traci.route.getEdges(traci.vehicle.getRouteID(veh_id))
  return veh_state_dict

def get_lanelet_graph(sumo_net_xml_file):
  """
  Create the lanelet graph from net.xml file using sumolib.
  lanelet ids are the vertices of the graph and lanelet connections are the edges
  Args:
    file path of net.xml file
  Returns:
    (lanelet_id_list, connection_list)
    lanelet_id_list:
      [lanelet_id_0, lanelet_id_1...]
    connection_list:
      [{"src_lanelet_id": src_lanelet_id, "dst_lanelet_id": dst_lanelet_id, "connection_type": "next"}, ...]
  """
  sumo_net = sumolib.net.readNet(sumo_net_xml_file)
  lanelet_id_list = []
  connection_list = []
  sumo_edges = sumo_net.getEdges()
  
  for sumo_edge in sumo_edges:
    # add "next" and "previous" connection
    for sumo_lane in sumo_edge.getLanes():
      lanelet_id_list += [sumo_lane.getID()]
      for sumo_connection in sumo_lane.getOutgoing():
        connection = {}
        connection["src_lanelet_id"] = sumo_lane.getID()
        connection["dst_lanelet_id"] = sumo_connection.getToLane().getID()
        connection["connection_type"] = "next"
        connection_list += [connection.copy()]
        # add the reverse connection
        connection["src_lanelet_id"] = connection["dst_lanelet_id"]
        connection["dst_lanelet_id"] = sumo_lane.getID()
        connection["connection_type"] = "previous"
        connection_list += [connection.copy()]
    # add "left" and "right" connection     
    for i in range (0, len(sumo_edge.getLanes()) - 1):
      connection = {}
      connection["src_lanelet_id"] = sumo_edge.getLanes()[i].getID()
      connection["dst_lanelet_id"] = sumo_edge.getLanes()[i+1].getID()
      connection["connection_type"] = "left"
      connection_list += [connection.copy()]
      connection["src_lanelet_id"] = sumo_edge.getLanes()[i+1].getID()
      connection["dst_lanelet_id"] = sumo_edge.getLanes()[i].getID()
      connection["connection_type"] = "right"
      connection_list += [connection.copy()]
  #"left" and "right" connection for opposite direction lane is not added
  return (lanelet_id_list, connection_list)
 
