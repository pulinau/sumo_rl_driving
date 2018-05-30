#!python3
__author__ = "Changjian Li"

import heapq

from include import *

def get_observation_space(env):
  observation_space = spaces.Dict({"ego_speed": spaces.Box(0, env.MAX_VEH_SPEED, shape=(1,), dtype=np.float32),
             "ego_dist_to_end_of_lane": spaces.Box(0, env.OBSERVATION_RADIUS, shape=(1,), dtype=np.float32),
             "ego_in_intersection": spaces.Discrete(2),
             "ego_exists_left_lane": spaces.Discrete(2),
             "ego_exists_right_lane": spaces.Discrete(2),
             "ego_correct_lane_gap": spaces.Box(-env.NUM_LANE_CONSIDERED, env.NUM_LANE_CONSIDERED, shape=(1,), dtype=np.int16),
             "exists_vehicle": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "speed": spaces.Box(0, env.MAX_VEH_SPEED, (env.NUM_VEH_CONSIDERED,), dtype=np.float32),  # absolute speed
             "dist_to_end_of_lane": spaces.Box(0, env.OBSERVATION_RADIUS, (env.NUM_VEH_CONSIDERED,), dtype=np.float32),
             "in_intersection": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "relative_position": spaces.Box(-env.OBSERVATION_RADIUS, env.OBSERVATION_RADIUS, (env.NUM_VEH_CONSIDERED, 2), dtype=np.float32), 
             "relative_heading": spaces.Box(-np.pi, np.pi, (env.NUM_VEH_CONSIDERED,), dtype=np.float32), # anti-clockwise
             "has_priority": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_peer": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_conflict": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_next": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_prev": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_left": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_right": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_ahead": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_behind": spaces.MultiBinary(env.NUM_VEH_CONSIDERED)
             })
  return observation_space

def get_veh_dict(env):
  """get the current state of all vehicles."""
  veh_id_list = env.tc.vehicle.getIDList()
  veh_dict = {}
  
  for veh_id in veh_id_list:
    veh_dict[veh_id] = {}
    veh_dict[veh_id]["type"] = "vehicle"
    veh_dict[veh_id]["position"] = env.tc.vehicle.getPosition(veh_id)
    veh_dict[veh_id]["angle"] = env.tc.vehicle.getAngle(veh_id) # in degree. North is zero, clockwise
    veh_dict[veh_id]["speed"] = env.tc.vehicle.getSpeed(veh_id)
    veh_dict[veh_id]["dimension"] = (env.tc.vehicle.getLength(veh_id), env.tc.vehicle.getWidth(veh_id))
    veh_dict[veh_id]["edge_id"] = env.tc.vehicle.getRoadID(veh_id)
    veh_dict[veh_id]["lane_id"] = env.tc.vehicle.getLaneID(veh_id)
    veh_dict[veh_id]["lane_index"] = env.tc.vehicle.getLaneIndex(veh_id)
    veh_dict[veh_id]["lane_length"] = env.tc.lane.getLength(veh_dict[veh_id]["lane_id"]) 
    veh_dict[veh_id]["lane_position"] = env.tc.vehicle.getLanePosition(veh_id) # position in the lane
    veh_dict[veh_id]["route"] = env.tc.vehicle.getRoute(veh_id)
    
    route = veh_dict[veh_id]["route"] # route is an edge id list of the vehicle's route
    if len(route) > env.tc.vehicle.getRouteIndex(veh_id) + 1: 
      veh_dict[veh_id]["next_normal_edge_id"] = route[env.tc.vehicle.getRouteIndex(veh_id) + 1]
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
        lanelet_dict[lane_id]["edge_priority"] = np.float("inf")
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
      edge_dict[edge_id]["priority"] = np.float("inf")
    else:
      edge_dict[edge_id]["priority"] = edge.getPriority()
    
  return edge_dict

def get_obs_dict(env):
  
  veh_dict = get_veh_dict(env)
  lanelet_dict = env.lanelet_dict
  edge_dict = env.edge_dict
  
  obs_dict = {}
  
  ego_dict = veh_dict[env.EGO_VEH_ID]
  lane_id_list_ego_edge = edge_dict[ego_dict["edge_id"]]["lane_id_list"]
  if ego_dict["next_normal_edge_id"] != None:
    lane_id_list_ego_next_normal_edge = edge_dict[ego_dict["next_normal_edge_id"]]["lane_id_list"]
  else:
    lane_id_list_ego_next_normal_edge = []
    
  obs_dict["ego_speed"] = ego_dict["speed"]
  obs_dict["ego_dist_to_end_of_lane"] = min(ego_dict["lane_length"] - ego_dict["lane_position"], env.OBSERVATION_RADIUS)
  
  # lanes inside intersections have ids that start with ":"
  if ego_dict["lane_id"][0] == ":":
    obs_dict["ego_in_intersection"] = 1
  else:
    obs_dict["ego_in_intersection"] = 0

  # couldChangeLane has a time lag of one step, a workaround is needed until this is fixed
  #if env.tc.vehicle.couldChangeLane(env.EGO_VEH_ID, 1):
  if ego_dict["lane_index"] < len(lane_id_list_ego_edge)-1:
    obs_dict["ego_exists_left_lane"] = 1
  else:
    obs_dict["ego_exists_left_lane"] = 0
  if ego_dict["lane_index"] != 0:
    obs_dict["ego_exists_right_lane"] = 1
  else:
    obs_dict["ego_exists_right_lane"] = 0
  
  # correct lane
  # if next normal edge doesn't exist, consider ego to be already in correct lane
  obs_dict["ego_correct_lane_gap"] = 0
  min_lane_gap = None
  for x in lane_id_list_ego_next_normal_edge:
    for y in lane_id_list_ego_edge:
      if internal_lane_id_between_lanes(y, x, lanelet_dict) != None:
        lane_gap = lanelet_dict[y]["lane_index"] - ego_dict["lane_index"]
        if min_lane_gap is None or abs(lane_gap) < min_lane_gap:
          min_lane_gap = lane_gap
  if min_lane_gap == None:
    obs_dict["ego_correct_lane_gap"] = 0
  else:
    obs_dict["ego_correct_lane_gap"] = min_lane_gap

  if obs_dict["ego_correct_lane_gap"] > 0:
    obs_dict["ego_correct_lane_gap"] = min(obs_dict["ego_correct_lane_gap"], env.NUM_LANE_CONSIDERED)
  else:
    obs_dict["ego_correct_lane_gap"] = max(obs_dict["ego_correct_lane_gap"], -env.NUM_LANE_CONSIDERED)
  
  # vehicles inside region of insterest
  def in_ROI(ego_position, veh_position):
    if ((veh_position[0] > ego_position[0]-env.OBSERVATION_RADIUS) and 
        (veh_position[1] > ego_position[1]-env.OBSERVATION_RADIUS) and 
        (veh_position[0] < ego_position[0]+env.OBSERVATION_RADIUS) and 
        (veh_position[1] < ego_position[1]+env.OBSERVATION_RADIUS)
        ):
      return True
    return False
  veh_id_list_ROI = [k for k, v in veh_dict.items() if k!=env.EGO_VEH_ID and in_ROI(ego_dict["position"], v["position"])]

  # now deal with the relavant vehicles
  obs_dict["exists_vehicle"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["speed"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["dist_to_end_of_lane"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["in_intersection"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["relative_position"] = [[0, 0]] * env.NUM_VEH_CONSIDERED
  obs_dict["relative_heading"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["has_priority"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_peer"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_conflict"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_next"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_prev"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_left"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_right"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_ahead"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_behind"] = [0] * env.NUM_VEH_CONSIDERED
  obs_dict["veh_relation_irrelevant"] = [0] * env.NUM_VEH_CONSIDERED

  # sort veh within ROI by distance to ego
  veh_heap = []
  for veh_id in veh_id_list_ROI:
    state_dict = veh_dict[veh_id]
    heapq.heappush(veh_heap, (np.linalg.norm(np.array(state_dict["position"]) - np.array(ego_dict["position"])), veh_id))
  
  for veh_index in range(min(env.NUM_VEH_CONSIDERED, len(veh_heap))):
    _, veh_id = heapq.heappop(veh_heap)
    state_dict = veh_dict[veh_id]
    
    obs_dict["exists_vehicle"][veh_index] = 1
    obs_dict["speed"][veh_index] = state_dict["speed"]
    obs_dict["dist_to_end_of_lane"][veh_index] = min(state_dict["lane_length"] - state_dict["lane_position"], env.OBSERVATION_RADIUS)
    
    if state_dict["edge_id"][0] == ':':
      obs_dict["in_intersection"][veh_index] = 1
    
    # transform the position to ego coordinate
    ego_angle_rad = ego_dict["angle"]/180 * np.pi
    rotation_mat = np.array([[np.cos(ego_angle_rad), -np.sin(ego_angle_rad)],
                             [np.sin(ego_angle_rad), np.cos(ego_angle_rad)]])
    relative_position = np.array(state_dict["position"]) - np.array(ego_dict["position"])
    relative_position = np.matmul(rotation_mat, relative_position)
    obs_dict["relative_position"][veh_index] = relative_position
    
    relative_heading = -(state_dict["angle"] - ego_dict["angle"])/180 * np.pi # anti-clockwise
    if relative_heading > np.pi:
      relative_heading -= 2*np.pi
    elif relative_heading < -np.pi:
      relative_heading += 2*np.pi
    obs_dict["relative_heading"][veh_index] = relative_heading
    
    # vehicle has priority over ego if the vehicle is
    # approaching/in the same intersection and it's inside a lane of higher priority
    # note that intersections (internal edges) are assigned the highest priority 
    if edge_dict[state_dict["edge_id"]]["to_node_id"] == edge_dict[ego_dict["edge_id"]]["to_node_id"] and \
       edge_dict[state_dict["edge_id"]]["priority"] > edge_dict[ego_dict["edge_id"]]["priority"]:
      obs_dict["has_priority"][veh_index] = 1
    
    # check if the each of the possible relationship holds for the vehicle
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
            obs_dict["veh_relation_peer"][veh_index] = 1 # PEER
    
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
                  obs_dict["veh_relation_conflict"][veh_index] = 1 # CONFLICT
    
    # NEXT, PREV
    if state_dict["lane_id"] in lanelet_dict[ego_dict["lane_id"]]["next_lane_id_list"]:
      obs_dict["veh_relation_next"][veh_index] = 1 # NEXT
    if ego_dict["lane_id"] in lanelet_dict[state_dict["lane_id"]]["next_lane_id_list"]:
      obs_dict["veh_relation_prev"][veh_index] = 1 # PREV
    
    # LEFT, RIGHT
    if state_dict["lane_id"] == lanelet_dict[ego_dict["lane_id"]]["left_lane_id"]:
      obs_dict["veh_relation_left"][veh_index] = 1 # LEFT
    if state_dict["lane_id"] == lanelet_dict[ego_dict["lane_id"]]["right_lane_id"]:
      obs_dict["veh_relation_right"][veh_index] = 1 # RIGHT
    
    # AHEAD, BEHIND
    if state_dict["lane_id"] == ego_dict["lane_id"]:
      if state_dict["lane_position"] > ego_dict["lane_position"]:
        obs_dict["veh_relation_ahead"][veh_index] = 1 # AHEAD
      else:
        obs_dict["veh_relation_behind"][veh_index] = 1 # BEHIND
  
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
  return False

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
