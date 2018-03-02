#!python3
__author__ = "Changjian Li"

from action import *
from observation import *
from sumo_gym import *

import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv1D, Lambda
from keras.models import Model, Sequential
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
            "--collision.action", "none", 
            "--collision.check-junctions", "true", 
            "--step-length", str(SUMO_TIME_STEP)]
#            "--lanechange.duration", "2"]
NET_XML_FILE = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.net.xml"

EGO_VEH_ID = "ego"
MAX_VEH_ACCEL = 2.6
MAX_VEH_DECEL = 4.5
MAX_VEH_SPEED = 55.55

# --------------------------
#        observation
# --------------------------
NUM_LANE_CONSIDERED = 3 # number of lanes considered on each side of ego
NUM_VEH_CONSIDERED = 16
OBSERVATION_RADIUS = 600

# --------------------------
#         reward
# --------------------------
MAX_COMFORT_ACCEL = 2.0
MAX_COMFORT_DECEL = 2.0

def build_model_safety(sumo_cfg, dqn_cfg):
  ego_input = Input(shape=(4, ))
  env_input = Input(shape=(10*sumo_cfg.NUM_VEH_CONSIDERED, 1))
  l1_0 = Dense(16, activation = None)(ego_input)
  l1_1 = Conv1D(16, kernel_size = 10, 
                strides = 10, padding = 'valid', 
                activation = None)(env_input)
  l1_1 = Lambda(lambda x: tf.reduce_sum(x, axis=1))(l1_1)
  l1 = keras.layers.add([l1_0, l1_1])
  l1 = keras.layers.Activation(activation="relu")(l1)
  l2 = Dense(16, activation='sigmoid')(l1)
  l3 = Dense(16, activation='sigmoid')(l2)
  y = Dense(dqn_cfg.action_size, activation='linear')(l3)
  model = Model(inputs = [ego_input, env_input], outputs=y)
  model.compile(loss='mse',
                optimizer=Adam(lr=0.001))
  return model

def reshape_safety(sumo_cfg, obs_dict):
  """reshape gym observation to keras neural network input"""
  out0 = np.array([], dtype = np.float32)
  out0  = np.append(out0, np.array(obs_dict["ego_speed"])/sumo_cfg.MAX_VEH_SPEED)
  out0  = np.append(out0, np.array(obs_dict["ego_dist_to_end_of_lane"])/sumo_cfg.OBSERVATION_RADIUS)
  out0  = np.append(out0, np.array(obs_dict["ego_exists_left_lane"]))
  out0  = np.append(out0, np.array(obs_dict["ego_exists_right_lane"]))
  out1 = np.reshape(np.array([], dtype = np.float32), (0, sumo_cfg.NUM_VEH_CONSIDERED))
  out1  = np.append(out1, np.array([obs_dict["exists_vehicle"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["speed"]])/sumo_cfg.MAX_VEH_SPEED, axis=0)
  out1  = np.append(out1, np.array([obs_dict["dist_to_end_of_lane"]])/sumo_cfg.OBSERVATION_RADIUS, axis=0)
  out1  = np.append(out1, np.array(obs_dict["relative_position"]).T/sumo_cfg.OBSERVATION_RADIUS, axis=0)
  out1  = np.append(out1, np.array([obs_dict["relative_heading"]])/pi, axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_left"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_right"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_ahead"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_behind"]]), axis=0)
  return [np.reshape(out0, (1,) + out0.shape), np.reshape(out1.T, (1, -1, 1))]

def build_model_regulation(sumo_cfg, dqn_cfg):
  ego_input = Input(shape=(6 + 2*sumo_cfg.NUM_LANE_CONSIDERED, ))
  env_input = Input(shape=(16*sumo_cfg.NUM_VEH_CONSIDERED, 1))
  l1_0 = Dense(24, activation = None)(ego_input)
  l1_1 = Conv1D(24, kernel_size = 16, strides = 16, padding = 'valid', 
                activation = None)(env_input)
  l1_1 = Lambda(lambda x: tf.reduce_sum(x, axis=1))(l1_1)
  l1 = keras.layers.add([l1_0, l1_1])
  l1 = keras.layers.Activation(activation="relu")(l1)
  l2 = Dense(16, activation='sigmoid')(l1)
  l3 = Dense(16, activation='sigmoid')(l2)
  y = Dense(dqn_cfg.action_size, activation='linear')(l3)
  model = Model(inputs = [ego_input, env_input], outputs=y)
  model.compile(loss='mse',
                optimizer=Adam(lr=0.001))
  return model

def reshape_regulation(sumo_cfg, obs_dict):
  out0 = np.array([], dtype = np.float32)
  out0 = np.append(out0, np.array(obs_dict["ego_speed"])/sumo_cfg.MAX_VEH_SPEED)
  out0 = np.append(out0, np.array(obs_dict["ego_dist_to_end_of_lane"])/sumo_cfg.OBSERVATION_RADIUS)
  out0 = np.append(out0, np.array(obs_dict["ego_in_intersection"]))
  out0 = np.append(out0, np.array(obs_dict["ego_exists_left_lane"]))
  out0 = np.append(out0, np.array(obs_dict["ego_exists_right_lane"]))
  out0 = np.append(out0, utils.to_categorical(obs_dict["ego_correct_lane_gap"] + sumo_cfg.NUM_LANE_CONSIDERED, 
                                      2*sumo_cfg.NUM_LANE_CONSIDERED + 1))
  out1 = np.reshape(np.array([], dtype = np.float32), (0, sumo_cfg.NUM_VEH_CONSIDERED))
  out1 = np.append(out1, np.array([obs_dict["exists_vehicle"]]), axis=0)
  out1 = np.append(out1, np.array([obs_dict["speed"]])/sumo_cfg.MAX_VEH_SPEED, axis=0)
  out1 = np.append(out1, np.array([obs_dict["dist_to_end_of_lane"]])/sumo_cfg.OBSERVATION_RADIUS, axis=0)
  out1 = np.append(out1, np.array([obs_dict["in_intersection"]]), axis=0)
  out1  = np.append(out1, np.array(obs_dict["relative_position"]).T/sumo_cfg.OBSERVATION_RADIUS, axis=0)
  out1  = np.append(out1, np.array([obs_dict["relative_heading"]])/pi, axis=0)
  out1  = np.append(out1, np.array([obs_dict["has_priority"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_peer"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_conflict"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_next"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_prev"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_left"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_right"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_ahead"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_behind"]]), axis=0)
  return [np.reshape(out0, (1, -1)), np.reshape(out1.T, (1, -1, 1))]

def build_model_comfort(sumo_cfg, dqn_cfg):
  model = Sequential()
  model.add(Dense(6, input_dim=dqn_cfg.state_size, activation='sigmoid'))
  model.add(Dense(6, activation='sigmoid'))
  model.add(Dense(dqn_cfg.action_size, activation='linear'))
  model.compile(loss='mse',
                optimizer=Adam(lr=0.001))
  return model

def reshape_comfort(sumo_cfg, obs_dict):
  return np.reshape(np.array([0], dtype = np.float32), (1, -1)) 

def build_model_speed(sumo_cfg, dqn_cfg):
  model = Sequential()
  model.add(Dense(6, input_dim=dqn_cfg.state_size, activation='sigmoid'))
  model.add(Dense(6, activation='sigmoid'))
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
                    memory_size = 64000, 
                    _build_model = build_model_safety, 
                    reshape = reshape_safety)

cfg_regulation = DQNCfg(name = "regulation", 
                        state_size = 6 + 2*NUM_LANE_CONSIDERED + 16*NUM_VEH_CONSIDERED, 
                        action_size = action_size, 
                        gamma = 0.99, 
                        epsilon = 0.2, 
                        threshold = -0.2, 
                        memory_size = 64000, 
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

agent_list = [DQNAgent(sumo_cfg, cfg_safety), DQNAgent(sumo_cfg, cfg_regulation), DQNAgent(sumo_cfg, cfg_comfort), DQNAgent(sumo_cfg, cfg_speed)]

# agent.load("./save/cartpole-dqn.h5")

env_state = EnvState.NORMAL
batch_size = 640

EPISODES = 10000

for e in range(EPISODES):
  obs_dict = env.reset()
  state_list = [agt.reshape(sumo_cfg, obs_dict) for agt in agent_list]
  
  for step in range(6400):
    #print("step: ", step, "agent_ctrl?;", env.agt_ctrl)
    action_set = set(range(action_size))
    for agt, state in zip(agent_list, state_list):
      action_set = agt.get_action_set(state, action_set)
    if len(action_set) >= 1:
      action = random.sample(action_set, 1)[0]
    else:
      print("*****************ERROR************************")
    
    next_obs_dict, reward_list, env_state, action_dict = env.step({"lane_change":ActionLaneChange(action//7), "accel_level":ActionAccel(action%7)})
    action = action_dict["lane_change"].value*7 + action_dict["accel_level"].value
    next_state_list = [agt.reshape(sumo_cfg, next_obs_dict) for agt in agent_list]
    
    for agt, state, reward, next_state in zip(agent_list, state_list, reward_list, next_state_list):
      agt.remember(state, action, reward, next_state, env_state)
      agt.learn(state, action, reward, next_state, env_state)
    
    #print("left?: ", obs_dict["ego_exists_left_lane"], "right?: ", obs_dict["ego_exists_right_lane"])
    #print(env.veh_dict_hist._history[0]["ego"])
    #print(action_dict, reward_list, env_state)
    
    
    
    if random.uniform(0, 1) < 0.1:
      if env.agt_ctrl == True:
        env.agt_ctrl = False
      else:
        env.agt_ctrl = True
    
    if env_state != EnvState.NORMAL:
      print("episode: {}/{}, step: {}"
            .format(e, EPISODES, step))
      print(action_dict, reward_list, env_state, env.agt_ctrl)
      break
    
    for agt in agent_list: 
      if len(agt.memory) > batch_size:
        agt.replay(batch_size)
      if e % 10 == 0:
        agt.save()
    obs_dict = next_obs_dict
    state_list = next_state_list
    
    #print("memory: ", agt.memory)
    #print("lane_change: ", ActionLaneChange(action//7), "accel_level: ", ActionAccel(action%7))
