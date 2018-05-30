#!python3
__author__ = "Changjian Li"

import numpy as np
import tensorflow as tf

from include import *
from sumo_cfgs import *
from dqn import DQNCfg

def reshape_validity(obs_dict):
  out = np.array([obs_dict["ego_exists_left_lane"], obs_dict["ego_exists_right_lane"]], dtype=np.int32)
  return [np.reshape(out, (1, -1))]

def select_actions_validity(state):
    ego_exists_left_lane = state[0][0][0]
    ego_exists_right_lane = state[0][0][1]

    if ego_exists_left_lane == 0 and ego_exists_right_lane == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
               ]
      sorted_idx = []
    if ego_exists_left_lane == 0 and ego_exists_right_lane == 1:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
               ]
      sorted_idx = []
    if ego_exists_left_lane == 1 and ego_exists_right_lane == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               ]
      sorted_idx = []
    if ego_exists_left_lane == 1 and ego_exists_right_lane == 1:
        valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                 ]
        sorted_idx = []

    return (set(valid), set([]), sorted_idx)

def reshape_safety(obs_dict):
  """reshape gym observation to keras neural network input"""
  o0 = np.array([obs_dict["ego_speed"]/MAX_VEH_SPEED,
                 min(obs_dict["ego_dist_to_end_of_lane"]/OBSERVATION_RADIUS, 1.0),
                 obs_dict["ego_in_intersection"],
                 obs_dict["ego_exists_left_lane"],
                 obs_dict["ego_exists_right_lane"]
                 ], dtype = np.float32)
  o1 = np.reshape(np.array([], dtype = np.float32), (0, NUM_VEH_CONSIDERED))
  o1  = np.append(o1, np.array([obs_dict["exists_vehicle"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["speed"]])/MAX_VEH_SPEED, axis=0)
  o1  = np.append(o1, np.minimum(np.array([obs_dict["dist_to_end_of_lane"]])/OBSERVATION_RADIUS,
                                     np.ones((1, NUM_VEH_CONSIDERED))), axis = 0)
  o1 = np.append(o1, np.array(obs_dict["relative_position"]).T / OBSERVATION_RADIUS, axis=0)
  o1  = np.append(o1, np.array([obs_dict["relative_heading"]])/np.pi, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_peer"]]), axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_conflict"]]), axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_next"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_left"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_right"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_ahead"]]), axis=0)

  o = [o0] + [x for x in o1.T]
  return [np.reshape(x, (1,) + x.shape) for x in o]

tf_cfg_safety = tf.ConfigProto()
tf_cfg_safety.gpu_options.per_process_gpu_memory_fraction = 0.4
#tf_cfg_safety = tf.ConfigProto(device_count = {"GPU": 0})

def build_model_safety():
  ego_input = tf.keras.layers.Input(shape=(5, ))
  ego_l1 = tf.keras.layers.Dense(12, activation=None)(ego_input)

  veh_inputs = [tf.keras.layers.Input(shape=(12,)) for _ in range(NUM_VEH_CONSIDERED)]
  shared_Dense1 = tf.keras.layers.Dense(10, activation=None)
  veh_l1 = [shared_Dense1(x) for x in veh_inputs]
  veh_l1 = [tf.keras.layers.add([ego_l1, x]) for x in veh_l1]
  veh_l1 = [tf.keras.layers.Activation(activation="sigmoid")(x) for x in veh_l1]
  shared_Dense2 = tf.keras.layers.Dense(32, activation=None)
  veh_l2 = [shared_Dense2(x) for x in veh_l1]

  l3 = tf.keras.layers.add(veh_l2)
  l3 = tf.keras.layers.Activation(activation="sigmoid")(l3)
  l4 = tf.keras.layers.Dense(32, activation="sigmoid")(l3)
  y = tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear')(l4)

  model = tf.keras.models.Model(inputs = [ego_input] + veh_inputs, outputs=y)
  opt = tf.keras.optimizers.RMSprop(lr=0.001)
  model.compile(loss='logcosh', optimizer=opt)
  return model

def reshape_regulation(obs_dict):
  lane_gap_1hot = [0] * (2*NUM_LANE_CONSIDERED + 1)
  lane_gap_1hot[obs_dict["ego_correct_lane_gap"] + NUM_LANE_CONSIDERED] = 1
  o0 = np.array([obs_dict["ego_speed"]/MAX_VEH_SPEED,
                   min(obs_dict["ego_dist_to_end_of_lane"] / OBSERVATION_RADIUS, 1.0),
                   obs_dict["ego_in_intersection"],
                   obs_dict["ego_exists_left_lane"],
                   obs_dict["ego_exists_right_lane"]
                   ] + lane_gap_1hot, dtype = np.float32)
  o1 = np.reshape(np.array([], dtype = np.float32), (0, NUM_VEH_CONSIDERED))
  o1 = np.append(o1, np.array([obs_dict["exists_vehicle"]]), axis=0)
  o1 = np.append(o1, np.array([obs_dict["speed"]])/MAX_VEH_SPEED, axis=0)
  o1  = np.append(o1, np.minimum(np.array([obs_dict["dist_to_end_of_lane"]])/OBSERVATION_RADIUS,
                                     np.ones((1, NUM_VEH_CONSIDERED))), axis = 0)
  o1 = np.append(o1, np.array([obs_dict["in_intersection"]]), axis=0)
  o1 = np.append(o1, np.array(obs_dict["relative_position"]).T / OBSERVATION_RADIUS, axis=0)
  o1  = np.append(o1, np.array([obs_dict["relative_heading"]])/np.pi, axis=0)
  o1  = np.append(o1, np.array([obs_dict["has_priority"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_peer"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_conflict"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_next"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_prev"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_left"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_right"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_ahead"]]), axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_behind"]]), axis=0)

  o = [o0] + [x for x in o1.T]
  return [np.reshape(x, (1,) + x.shape) for x in o]

tf_cfg_regulation = tf.ConfigProto()
tf_cfg_regulation.gpu_options.per_process_gpu_memory_fraction = 0.4
#tf_cfg_regulation = tf.ConfigProto(device_count = {"GPU": 0})

def build_model_regulation():
  ego_input = tf.keras.layers.Input(shape=(6 + 2*NUM_LANE_CONSIDERED, ))
  ego_l1 = tf.keras.layers.Dense(10, activation=None)(ego_input)

  veh_inputs = [tf.keras.layers.Input(shape=(16,)) for _ in range(NUM_VEH_CONSIDERED)]
  shared_Dense1 = tf.keras.layers.Dense(10, activation=None)
  veh_l1 = [shared_Dense1(x) for x in veh_inputs]
  veh_l1 = [tf.keras.layers.add([ego_l1, x]) for x in veh_l1]
  veh_l1 = [tf.keras.layers.Activation(activation="sigmoid")(x) for x in veh_l1]
  shared_Dense2 = tf.keras.layers.Dense(32, activation=None)
  veh_l2 = [shared_Dense2(x) for x in veh_l1]

  l3 = tf.keras.layers.add(veh_l2)
  l3 = tf.keras.layers.Activation(activation="sigmoid")(l3)
  l4 = tf.keras.layers.Dense(32, activation="sigmoid")(l3)
  y = tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear')(l4)

  model = tf.keras.models.Model(inputs = [ego_input] + veh_inputs, outputs=y)
  opt = tf.keras.optimizers.RMSprop(lr=0.001)
  model.compile(loss='logcosh', optimizer=opt)
  return model


def reshape_speed_comfort(obs_dict):
  out = np.array([obs_dict["ego_speed"], obs_dict["ego_correct_lane_gap"]], dtype = np.float32)
  return [np.reshape(out, (1,-1))]

def select_actions_speed_comfort(state):
  ego_speed = state[0][0][0]
  ego_correct_lane_gap = state[0][0][1]
  if ego_speed < MAX_VEH_SPEED-1.4:
    if ego_correct_lane_gap == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value]
      sorted_idx  = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    elif ego_correct_lane_gap > 0:
      valid = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value]
      sorted_idx  = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    else:
      valid = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value]
      sorted_idx  = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
  elif ego_speed > MAX_VEH_SPEED + 1.4:
    if ego_correct_lane_gap == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value
                     ]
    elif ego_correct_lane_gap > 0:
      valid = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    else:
      valid = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
  else:
    if ego_correct_lane_gap == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    elif ego_correct_lane_gap > 0:
      valid = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    elif ego_correct_lane_gap < 0:
      valid = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
  return (set(valid), set([]), sorted_idx)

action_size = len(ActionLaneChange) * len(ActionAccel)

cfg_validity = DQNCfg(name = "validity",
                      play=False,
                      state_size=2,
                      action_size=action_size,
                      pretrain_low_target=None,
                      pretrain_high_target=None,
                      gamma=None,
                      gamma_inc=None,
                      gamma_max=None,
                      epsilon=None,
                      epsilon_dec=None,
                      epsilon_min=None,
                      threshold=None,
                      memory_size=None,
                      traj_end_pred=None,
                      replay_batch_size=None,
                      _build_model=None,
                      tf_cfg=None,
                      reshape=reshape_validity,
                      _select_actions=select_actions_validity)

cfg_safety = DQNCfg(name = "safety",
                    play = False,
                    state_size = 4 + 10*NUM_VEH_CONSIDERED,
                    action_size = action_size,
                    pretrain_low_target=-10,
                    pretrain_high_target=0,
                    gamma = 0.9,
                    gamma_inc = 0.0005,
                    gamma_max = 0.99,
                    epsilon = 0.1,
                    epsilon_dec = 0.00001,
                    epsilon_min = 0.01,
                    threshold = -3,
                    memory_size = 6400,
                    traj_end_pred = lambda x: x < -0.1,
                    replay_batch_size = 640,
                    _build_model = build_model_safety,
                    tf_cfg = tf_cfg_safety,
                    reshape = reshape_safety)

cfg_regulation = DQNCfg(name = "regulation",
                        play = False,
                        state_size = 6 + 2*NUM_LANE_CONSIDERED + 16*NUM_VEH_CONSIDERED,
                        action_size = action_size,
                        pretrain_low_target=-10,
                        pretrain_high_target=0,
                        gamma = 0.9,
                        gamma_inc = 0.0005,
                        gamma_max = 0.99,
                        epsilon=0.1,
                        epsilon_dec=0.00001,
                        epsilon_min=0.02,
                        threshold = -8,
                        memory_size = 6400,
                        traj_end_pred = lambda x: x < -0.1,
                        replay_batch_size = 640,
                        _build_model = build_model_regulation,
                        tf_cfg = tf_cfg_regulation,
                        reshape = reshape_regulation)

cfg_speed_comfort = DQNCfg(name = "speed_comfort",
                           play = False,
                           state_size = 2,
                           action_size = action_size,
                           pretrain_low_target=None,
                           pretrain_high_target=None,
                           gamma = None,
                           gamma_inc = None,
                           gamma_max = None,
                           epsilon=None,
                           epsilon_dec=None,
                           epsilon_min=None,
                           threshold = None,
                           memory_size = None,
                           traj_end_pred = None,
                           replay_batch_size = None,
                           _build_model = None,
                           tf_cfg = None,
                           reshape = reshape_speed_comfort,
                           _select_actions=select_actions_speed_comfort)