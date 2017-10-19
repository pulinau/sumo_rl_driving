 #!python3

SUMO_TOOLS_DIR = "/home/ken/project/sumo-0.30.0/tools"
EGO_VEH_ID = "ego"

try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")

import traci
import sumolib


def get_vehicle_state():
  veh_id_list = traci.vehivle.getIDList()
  veh_state_dict = {}
  for veh_id in veh_id_list:
    veh_state_dict[veh_id] = {}
    veh_state_dict[veh_id]["type"] = "vehicle"
    veh_state_dict[vehicle]["location"] = traci.vehicle.getPosition(vehicle)
    veh_state_dict[vehicle]["velocity"] = traci.vehicle.getSpeed(vehicle)
    veh_state_dict[vehicle]["dimension"]  = [traci.vehicle.getLength(vehicle),traci.vehicle.getWidth(vehicle)]
    veh_state_dict[vehicle]["edge"] = traci.vehicle.getRoadID(vehicle)
    veh_state_dict[vehicle]["lane"] = traci.vehicle.getLaneID(vehicle)
    veh_state_dict[vehicle]["lane_length"] = traci.vehicle.getLaneLength(veh_state_dict[vehicle]["lane"])
    veh_state_dict[vehicle]["lane_position"] = traci.vehicle.getLanePosition(vehicle)
    veh_state_dict[vehicle]["route"] = troute.getEdges(traci.vehicle.getRouteID(vehicle))
  return veh_state_dict

def get_lanelet_graph(sumo_net_xml_file):
"""
Create the lanelet graph from net.xml file using sumolib
Args:
  file path net.xml
Returns:
  [{"src_lanelet_id": src_lanelet_id, "dst_lanelet_id": dst_lanelet_id, "connection_type": "next"}, ...]
"""
# lanelet ids are the vertices of the graph
# lanelet connections are the edges of the graph
  try:
    sumo_net = sumolib.readNet("sumo_net_xml_file")
  except:
    print("failed to open sumo net.xml file")
  
  lanelet_id_list = []
  connection_list = []
  sumo_edges = sumo_net.getEdges()
  
  for sumo_edge in sumo_edges:
    # add "next" and "previous" connection
    for sumo_lane in sumo_edge.getLanes():
      lanelet_id_list += [sumo_lane.getID()]
      for sumo_connection in sumo_lane.getOutgoing:
        connection = {}
        connection["src_lanelet_id"] = sumo_lane.getID()
        connection["dst_lanelet_id"] = sumo_connection.getToLane().getID()
        connection["connection_type"] = "next"
        connection_list += [connection]
        # add the reverse connection
        connection["src_lanelet_id"] = connection["dst_lanelet_id"]
        connection["dst_lanelet_id"] = sumo_lane.getID()
        connection["connection_type"] = "previous"
        connection_list += [connection]
    # add "left" and "right" connection     
    for i in range (0, len(sumo_edge.getLanes()) - 1):
      connection = {}
      connection["src_lanelet_id"] = sumo_edge.getLanes()[i].getID()
      connection["dst_lanelet_id"] = sumo_edge.getLanes()[i+1].getID()
      connection["connection_type"] = "left"
      connection_list += [connection]
      connection["src_lanelet_id"] = sumo_edge.getLanes()[i+1].getID()
      connection["dst_lanelet_id"] = sumo_edge.getLanes()[i].getID()
      connection["connection_type"] = "right"
      connection_list += [connection]
  # "left" and "right" connection for opposite direction lane is not added
  pass
