#!python3
__author__ = "Changjian Li"

from action import *
from observation import *
from sumo_gym import *

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import utils

from dqn import *
import random

# --------------------------
#          SUMO
# --------------------------
SUMO_TOOLS_DIR = "/home/ken/project/sumo-bin/tools"
SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo"
SUMO_CONFIG = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.sumocfg"
SUMO_TIME_STEP = 0.1
SUMO_CMD = [SUMO_BIN, "-c", SUMO_CONFIG, 
            "--time-to-teleport", "-1", 
            "--collision.action", "warn", 
            "--collision.check-junctions", "true", 
            "--step-length", str(SUMO_TIME_STEP)]
#            "--lanechange.duration", "2"]
NET_XML_FILE = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.net.xml"

EGO_VEH_ID = "ego"
MAX_VEH_ACCEL = 20
MAX_VEH_DECEL = 20
MAX_VEH_SPEED = 120

# --------------------------
#        observation
# --------------------------
NUM_LANE_CONSIDERED = 3 # number of lanes considered on each side of ego
NUM_VEH_CONSIDERED = 16
OBSERVATION_RADIUS = 600

# --------------------------
#         reward
# --------------------------
MAX_COMFORT_ACCEL = 10
MAX_COMFORT_DECEL = 10

def build_model_safety(dqn_cfg):
  model = Sequential()
  model.add(Dense(8, input_dim=dqn_cfg.state_size, activation='sigmoid'))
  model.add(Dense(8, activation='sigmoid'))
  model.add(Dense(dqn_cfg.action_size, activation='linear'))
  model.compile(loss='mse',
                optimizer=Adam(lr=0.001))
  return model

def reshape_safety(sumo_cfg, obs_dict):
  """reshape gym observation to keras neural network input"""
  out = np.array([], dtype = np.float32)
  out  = np.append(out, np.array(obs_dict["ego_speed"])/sumo_cfg.MAX_VEH_SPEED)
  out  = np.append(out, np.array(obs_dict["ego_dist_to_end_of_lane"])/sumo_cfg.OBSERVATION_RADIUS)
  out  = np.append(out, np.array(obs_dict["ego_exists_left_lane"]))
  out  = np.append(out, np.array(obs_dict["ego_exists_right_lane"]))
  out  = np.append(out, np.array(obs_dict["exists_vehicle"]))
  out  = np.append(out, np.array(obs_dict["speed"])/sumo_cfg.MAX_VEH_SPEED)
  out  = np.append(out, np.array(obs_dict["dist_to_end_of_lane"])/sumo_cfg.OBSERVATION_RADIUS)
  out  = np.append(out, np.reshape(np.array(obs_dict["relative_position"]), (-1, 1))/sumo_cfg.OBSERVATION_RADIUS)
  out  = np.append(out, np.array(obs_dict["relative_heading"])/pi)
  out  = np.append(out, np.array(obs_dict["veh_relation_left"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_right"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_ahead"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_behind"]))
  return np.reshape(out, (1, -1)) 

def build_model_regulation(dqn_cfg):
  model = Sequential()
  model.add(Dense(24, input_dim=dqn_cfg.state_size, activation='sigmoid'))
  model.add(Dense(24, activation='sigmoid'))
  model.add(Dense(dqn_cfg.action_size, activation='linear'))
  model.compile(loss='mse',
                optimizer=Adam(lr=0.001))
  return model

def reshape_regulation(sumo_cfg, obs_dict):
  out = np.array([], dtype = np.float32)
  out = np.append(out, np.array(obs_dict["ego_speed"])/sumo_cfg.MAX_VEH_SPEED)
  out = np.append(out, np.array(obs_dict["ego_dist_to_end_of_lane"])/sumo_cfg.OBSERVATION_RADIUS)
  out = np.append(out, np.array(obs_dict["ego_in_intersection"]))
  out = np.append(out, np.array(obs_dict["ego_exists_left_lane"]))
  out = np.append(out, np.array(obs_dict["ego_exists_right_lane"]))
  out = np.append(out, utils.to_categorical(obs_dict["ego_correct_lane_gap"] + sumo_cfg.NUM_LANE_CONSIDERED, 
                                      2*sumo_cfg.NUM_LANE_CONSIDERED + 1))
  out = np.append(out, np.array(obs_dict["exists_vehicle"]))
  out = np.append(out, np.array(obs_dict["speed"])/sumo_cfg.MAX_VEH_SPEED)
  out = np.append(out, np.array(obs_dict["dist_to_end_of_lane"])/sumo_cfg.OBSERVATION_RADIUS)
  out = np.append(out, np.array(obs_dict["in_intersection"]))
  out  = np.append(out, np.reshape(np.array(obs_dict["relative_position"]), (-1, 1))/env.OBSERVATION_RADIUS)
  out  = np.append(out, np.array(obs_dict["relative_heading"])/pi)
  out  = np.append(out, np.array(obs_dict["has_priority"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_peer"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_conflict"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_next"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_prev"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_left"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_right"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_ahead"]))
  out  = np.append(out, np.array(obs_dict["veh_relation_behind"]))
  return np.reshape(out, (1, -1)) 

def build_model_comfort(dqn_cfg):
  model = Sequential()
  model.add(Dense(8, input_dim=dqn_cfg.state_size, activation='sigmoid'))
  model.add(Dense(8, activation='sigmoid'))
  model.add(Dense(dqn_cfg.action_size, activation='linear'))
  model.compile(loss='mse',
                optimizer=Adam(lr=0.001))
  return model

def reshape_comfort(sumo_cfg, obs_dict):
  return np.reshape(np.array([0], dtype = np.float32), (1, -1)) 

def build_model_speed(dqn_cfg):
  model = Sequential()
  model.add(Dense(8, input_dim=dqn_cfg.state_size, activation='sigmoid'))
  model.add(Dense(8, activation='sigmoid'))
  model.add(Dense(dqn_cfg.action_size, activation='linear'))
  model.compile(loss='mse',
                optimizer=Adam(lr=0.001))
  return model

def reshape_speed(sumo_cfg, obs_dict):
  return np.reshape(np.array(obs_dict["ego_speed"]/sumo_cfg.MAX_VEH_SPEED, dtype = np.float32), (1, -1)) 

action_size = len(ActionLaneChange) * len(ActionAccel)

cfg_safety = DQNCfg(name = "safety", 
                    state_size = 4 + 10*NUM_VEH_CONSIDERED, 
                    action_size = action_size, 
                    gamma = 0.99, 
                    epsilon = 0.2, 
                    threshold = -0.1, 
                    memory_size = 6400, 
                    _build_model = build_model_safety, 
                    reshape = reshape_safety)

cfg_regulation = DQNCfg(name = "regulation", 
                        state_size = 6 + 2*NUM_LANE_CONSIDERED + 16*NUM_VEH_CONSIDERED, 
                        action_size = action_size, 
                        gamma = 0.99, 
                        epsilon = 0.2, 
                        threshold = -0.2, 
                        memory_size = 6400, 
                        _build_model = build_model_regulation, 
                        reshape = reshape_regulation)

cfg_comfort = DQNCfg(name = "comfort", 
                     state_size = 1, 
                     action_size = action_size, 
                     gamma = 0, 
                     epsilon = 0.2, 
                     threshold = -0.4, 
                     memory_size = 640, 
                     _build_model = build_model_comfort, 
                     reshape = reshape_comfort)

cfg_speed = DQNCfg(name = "speed", 
                   state_size = 1, 
                   action_size = action_size, 
                   gamma = 0, 
                   epsilon = 0.1, 
                   threshold = -0.4, 
                   memory_size = 640, 
                   _build_model = build_model_speed, 
                   reshape = reshape_speed)

sumo_cfg = SumoCfg(
               # sumo
               SUMO_CMD, 
               SUMO_TIME_STEP, 
               NET_XML_FILE, 
               EGO_VEH_ID, 
               MAX_VEH_ACCEL, 
               MAX_VEH_DECEL, 
               MAX_VEH_SPEED, 
               # observation
               NUM_LANE_CONSIDERED, 
               NUM_VEH_CONSIDERED, 
               OBSERVATION_RADIUS, 
               # reward
               MAX_COMFORT_ACCEL, 
               MAX_COMFORT_DECEL)

env = MultiObjSumoEnv(sumo_cfg)

agent_list = [DQNAgent(cfg_safety), DQNAgent(cfg_regulation), DQNAgent(cfg_comfort), DQNAgent(cfg_speed)]

# agent.load("./save/cartpole-dqn.h5")

env_state = EnvState.NORMAL
batch_size = 32

for e in range(EPISODES):
  obs_dict = env.reset()
  state_list = [agt.reshape(sumo_cfg, obs_dict) for agt in agent_list]
  
  
  for step in range(6400):
    action_set = set(range(action_size))
    for agt, state in zip(agent_list, state_list):
      action_set = agt.get_action_set(state, action_set)
    action = random.sample(action_set, 1)[0]
    next_obs_dict, reward_list, env_state, _ = env.step({"lane_change":ActionLaneChange(action//7), "accel_level":ActionAccel(action%7)})
    next_state_list = [agt.reshape(sumo_cfg, next_obs_dict) for agt in agent_list]
    for agt, state, reward, next_state in zip(agent_list, state_list, reward_list, next_state_list):
      agt.remember(state, action, reward, next_state, env_state)
    
    if env_state != EnvState.NORMAL:
      print("episode: {}/{}, step: {}"
            .format(e, EPISODES, step))
      break
    
    for agt in agent_list:  
      if len(agt.memory) > batch_size:
        agt.replay(batch_size)
      if e % 10 == 0:
        agt.save()
    state_list = next_state_list

