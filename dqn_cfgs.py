import numpy as np
import tensorflow as tf

from include import *
from sumo_cfgs import *
from dqn import DQNCfg
from math import pi


def reshape_safety(obs_dict):
  """reshape gym observation to keras neural network input"""
  out0 = np.array([], dtype = np.float32)
  out0  = np.append(out0, np.array(obs_dict["ego_speed"])/MAX_VEH_SPEED)
  out0  = np.append(out0, np.array(obs_dict["ego_dist_to_end_of_lane"])/OBSERVATION_RADIUS)
  out0  = np.append(out0, np.array(obs_dict["ego_exists_left_lane"]))
  out0  = np.append(out0, np.array(obs_dict["ego_exists_right_lane"]))
  out1 = np.reshape(np.array([], dtype = np.float32), (0, NUM_VEH_CONSIDERED))
  out1  = np.append(out1, np.array([obs_dict["exists_vehicle"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["speed"]])/MAX_VEH_SPEED, axis=0)
  out1  = np.append(out1, np.array([obs_dict["dist_to_end_of_lane"]])/OBSERVATION_RADIUS, axis=0)
  out1  = np.append(out1, np.array(obs_dict["relative_position"]).T/OBSERVATION_RADIUS, axis=0)
  out1  = np.append(out1, np.array([obs_dict["relative_heading"]])/pi, axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_left"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_right"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_ahead"]]), axis=0)
  out1  = np.append(out1, np.array([obs_dict["veh_relation_behind"]]), axis=0)
  return [np.reshape(out0, (1,) + out0.shape), np.reshape(out1.T, (1, -1, 1, 1))]

tf_cfg_safety = tf.ConfigProto()
tf_cfg_safety.gpu_options.per_process_gpu_memory_fraction = 0.4

def build_model_safety():
  ego_input = tf.keras.layers.Input(shape=(4, ))
  env_input = tf.keras.layers.Input(shape=(10* NUM_VEH_CONSIDERED, 1, 1))
  l1_0 = tf.keras.layers.Dense(64, activation = None)(ego_input)
  l1_1 = tf.keras.layers.Conv2D(64, kernel_size = (10, 1),
                                strides = (10, 1), padding = 'valid',
                                activation = None)(env_input)
  l1_1 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.reshape(x, [-1, NUM_VEH_CONSIDERED, 64]), axis=1))(l1_1)
  l1 = tf.keras.layers.add([l1_0, l1_1])
  l1 = tf.keras.layers.Activation(activation="relu")(l1)
  l2 = tf.keras.layers.Dense(64, activation=None)(l1)
  l2 = tf.keras.layers.BatchNormalization()(l2)
  l2 = tf.keras.layers.Activation('sigmoid')(l2)
  l3 = tf.keras.layers.Dense(64, activation=None)(l2)
  l3 = tf.keras.layers.BatchNormalization()(l3)
  l3 = tf.keras.layers.Activation('sigmoid')(l3)
  y = tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear')(l3)
  model = tf.keras.models.Model(inputs = [ego_input, env_input], outputs=y)
  model.compile(loss='mse', optimizer="SGD")
  return model

def reshape_regulation(obs_dict):
  out0 = np.array([], dtype = np.float32)
  out0 = np.append(out0, np.array(obs_dict["ego_speed"])/MAX_VEH_SPEED)
  out0 = np.append(out0, np.array(obs_dict["ego_dist_to_end_of_lane"])/OBSERVATION_RADIUS)
  out0 = np.append(out0, np.array(obs_dict["ego_in_intersection"]))
  out0 = np.append(out0, np.array(obs_dict["ego_exists_left_lane"]))
  out0 = np.append(out0, np.array(obs_dict["ego_exists_right_lane"]))
  lane_gap_1hot = np.array([0] * (2*NUM_LANE_CONSIDERED + 1))
  lane_gap_1hot[obs_dict["ego_correct_lane_gap"] + NUM_LANE_CONSIDERED] = 1
  out0 = np.append(out0, lane_gap_1hot)
  out1 = np.reshape(np.array([], dtype = np.float32), (0, NUM_VEH_CONSIDERED))
  out1 = np.append(out1, np.array([obs_dict["exists_vehicle"]]), axis=0)
  out1 = np.append(out1, np.array([obs_dict["speed"]])/MAX_VEH_SPEED, axis=0)
  out1 = np.append(out1, np.array([obs_dict["dist_to_end_of_lane"]])/OBSERVATION_RADIUS, axis=0)
  out1 = np.append(out1, np.array([obs_dict["in_intersection"]]), axis=0)
  out1  = np.append(out1, np.array(obs_dict["relative_position"]).T/OBSERVATION_RADIUS, axis=0)
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
  return [np.reshape(out0, (1, -1)), np.reshape(out1.T, (1, -1, 1, 1))]

tf_cfg_regulation = tf.ConfigProto()
tf_cfg_regulation.gpu_options.per_process_gpu_memory_fraction = 0.4

def build_model_regulation():
  ego_input = tf.keras.layers.Input(shape=(6 + 2*NUM_LANE_CONSIDERED, ))
  env_input = tf.keras.layers.Input(shape=(16*NUM_VEH_CONSIDERED, 1, 1))
  l1_0 = tf.keras.layers.Dense(64, activation = None)(ego_input)
  l1_1 = tf.keras.layers.Conv2D(64, kernel_size = (16, 1), strides = (16, 1), padding = 'valid',
                activation = None)(env_input)
  l1_1 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.reshape(x, [-1, NUM_VEH_CONSIDERED, 64]), axis=1))(l1_1)
  l1 = tf.keras.layers.add([l1_0, l1_1])
  l1 = tf.keras.layers.Activation(activation="relu")(l1)
  l2 = tf.keras.layers.Dense(64, activation=None)(l1)
  l2 = tf.keras.layers.BatchNormalization()(l2)
  l2 = tf.keras.layers.Activation('sigmoid')(l2)
  l3 = tf.keras.layers.Dense(64, activation=None)(l2)
  l3 = tf.keras.layers.BatchNormalization()(l3)
  l3 = tf.keras.layers.Activation('sigmoid')(l3)
  y = tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear')(l3)
  model = tf.keras.models.Model(inputs = [ego_input, env_input], outputs=y)
  model.compile(loss='mse', optimizer="SGD")
  return model

def reshape_comfort(obs_dict):
  return np.reshape(np.array([0], dtype = np.float32), (1, -1))

tf_cfg_comfort = tf.ConfigProto(device_count = {"GPU": 0})

def build_model_comfort():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(8, input_dim=1, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear'))
  model.compile(loss='mse', optimizer="SGD")
  return model

def reshape_speed(obs_dict):
  return np.reshape(np.array(obs_dict["ego_speed"]/MAX_VEH_SPEED, dtype = np.float32), (1, -1))

tf_cfg_speed = tf.ConfigProto(device_count = {"GPU": 0})

def build_model_speed():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(8, input_dim=1, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear'))
  model.compile(loss='mse', optimizer="SGD")
  return model

action_size = len(ActionLaneChange) * len(ActionAccel)

cfg_safety = DQNCfg(name = "safety",
                    play = False,
                    state_size = 4 + 10*NUM_VEH_CONSIDERED,
                    action_size = action_size,
                    gamma = 0,
                    gamma_inc = 0.00005,
                    gamma_max = 0.90,
                    epsilon = 0.2,
                    threshold = -5,
                    memory_size = 640000,
                    replay_batch_size = 1600,
                    _build_model = build_model_safety,
                    tf_cfg = tf_cfg_safety,
                    reshape = reshape_safety)

cfg_regulation = DQNCfg(name = "regulation",
                        play = False,
                        state_size = 6 + 2*NUM_LANE_CONSIDERED + 16*NUM_VEH_CONSIDERED,
                        action_size = action_size,
                        gamma = 0,
                        gamma_inc = 0.00005,
                        gamma_max = 0.90,
                        epsilon = 0.2,
                        threshold = -5,
                        memory_size = 640000,
                        replay_batch_size = 1600,
                        _build_model = build_model_regulation,
                        tf_cfg = tf_cfg_regulation,
                        reshape = reshape_regulation)

cfg_comfort = DQNCfg(name = "comfort",
                     play = False,
                     state_size = 1,
                     action_size = action_size,
                     gamma = 0,
                     gamma_inc = 0,
                     gamma_max = 0,
                     epsilon = 0.5,
                     threshold = -8,
                     memory_size = 640,
                     replay_batch_size = 64,
                     _build_model = build_model_comfort,
                     tf_cfg = tf_cfg_comfort,
                     reshape = reshape_comfort)

cfg_speed = DQNCfg(name = "speed",
                   play = False,
                   state_size = 1,
                   action_size = action_size,
                   gamma = 0,
                   gamma_inc = 0,
                   gamma_max = 0,
                   epsilon = 0.5,
                   threshold = -8,
                   memory_size = 640,
                   replay_batch_size = 64,
                   _build_model = build_model_speed,
                   tf_cfg = tf_cfg_speed,
                   reshape = reshape_speed)