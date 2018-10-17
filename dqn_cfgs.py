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
      sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                    ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                    ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                    ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                    ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                    ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                    ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                    7, 8]
    if ego_exists_left_lane == 0 and ego_exists_right_lane == 1:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               8]
      sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 8, 7]
    if ego_exists_left_lane == 1 and ego_exists_right_lane == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               7]
      sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 7, 8]
    if ego_exists_left_lane == 1 and ego_exists_right_lane == 1:
        valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 7, 8]
        sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 7, 8]

    return (set(valid), sorted_idx)

def reshape_safety(obs_dict):
  """reshape gym observation to keras neural network input"""
  # sqrt is used to strech the input to emphasize the near zero part
  o0 = np.array([np.sqrt(obs_dict["ego_speed"]/MAX_VEH_SPEED) - 0.5,
                 np.sqrt(min(obs_dict["ego_dist_to_end_of_lane"]/OBSERVATION_RADIUS, 1.0)) - 0.5,
                 obs_dict["ego_in_intersection"] - 0.5,
                 obs_dict["ego_exists_left_lane"] - 0.5,
                 obs_dict["ego_exists_right_lane"] - 0.5
                 ], dtype = np.float32)
  o1 = np.reshape(np.array([], dtype = np.float32), (0, NUM_VEH_CONSIDERED))
  o1  = np.append(o1, np.array([obs_dict["exists_vehicle"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["brake_signal"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["left_signal"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["right_signal"]]) - 0.5, axis=0)
  rel_speed = np.array([obs_dict["relative_speed"]]) / MAX_VEH_SPEED + 0.5
  rel_speed = np.minimum(np.sqrt(np.abs(rel_speed)), np.ones((1, NUM_VEH_CONSIDERED))*0.5) * np.sign(rel_speed)
  o1  = np.append(o1, rel_speed , axis=0)
  o1  = np.append(o1, np.sqrt(np.minimum(np.array([obs_dict["dist_to_end_of_lane"]])/OBSERVATION_RADIUS,
                              np.ones((1, NUM_VEH_CONSIDERED)))) - 0.5, axis = 0)
  rel_pos = np.array(obs_dict["relative_position"]).T / 2 * OBSERVATION_RADIUS
  rel_pos = np.sqrt(np.abs(rel_pos)) * np.sign(rel_pos)
  o1 = np.append(o1, rel_pos, axis=0)
  o1  = np.append(o1, np.array([obs_dict["relative_heading"]])/2*np.pi, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_peer"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_conflict"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_left"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_right"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_ahead"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_behind"]]) - 0.5, axis=0)
  ttc = np.array([obs_dict["ttc"]]) / MAX_TTC_CONSIDERED
  ttc = np.sqrt(np.abs(ttc)) * np.sign(ttc)
  o1 = np.append(o1, ttc - 0.5, axis=0)

  o = [o0] + [x for x in o1.T]
  return [[x] for x in o]

tf_cfg_safety = tf.ConfigProto()
tf_cfg_safety.gpu_options.per_process_gpu_memory_fraction = 0.4
#tf_cfg_safety = tf.ConfigProto(device_count = {"GPU": 0})

def build_model_safety():
  ego_input = tf.keras.layers.Input(shape=(5, ))
  ego_l1 = tf.keras.layers.Dense(64, activation=None)(ego_input)

  veh_inputs = [tf.keras.layers.Input(shape=(16,)) for _ in range(NUM_VEH_CONSIDERED)]
  shared_Dense1 = tf.keras.layers.Dense(64, activation=None)
  veh_l = [shared_Dense1(x) for x in veh_inputs]

  veh_l = [tf.keras.layers.add([ego_l1, x]) for x in veh_l]
  veh_l = [tf.keras.layers.Activation("sigmoid")(x) for x in veh_l]

  n_layers = 2
  Dense_list = [tf.keras.layers.Dense(64, activation=None) for _ in range(n_layers)]
  for i in range(n_layers):
    veh_l = [Dense_list[i](x) for x in veh_l]
    veh_l = [tf.keras.layers.Activation("sigmoid")(x) for x in veh_l]

  shared_Dense2 = tf.keras.layers.Dense(reduced_action_size, activation=None)
  veh_y = [shared_Dense2(x) for x in veh_l]

  y = tf.keras.layers.minimum(veh_y)

  model = tf.keras.models.Model(inputs=[ego_input] + veh_inputs, outputs=veh_y + [y])
  opt = tf.keras.optimizers.RMSprop(lr=0.0001)
  model.compile(loss='logcosh', optimizer=opt)

  return model

def reshape_regulation(obs_dict):
  lane_gap_1hot = [-0.5] * (2*NUM_LANE_CONSIDERED + 1)
  lane_gap_1hot[obs_dict["ego_correct_lane_gap"] + NUM_LANE_CONSIDERED] = 0.5

  o = np.array([np.sqrt(obs_dict["ego_speed"]/MAX_VEH_SPEED) - 0.5,
                np.sqrt(min(obs_dict["ego_dist_to_end_of_lane"] / OBSERVATION_RADIUS, 1.0)) - 0.5,
                obs_dict["ego_in_intersection"] - 0.5,
                obs_dict["ego_has_priority"] - 0.5,
                ] + lane_gap_1hot, dtype = np.float32)

  return [[o]]

tf_cfg_regulation = tf.ConfigProto()
tf_cfg_regulation.gpu_options.per_process_gpu_memory_fraction = 0.3

def build_model_regulation():
  x = tf.keras.layers.Input(shape=(5 + 2*NUM_LANE_CONSIDERED, ))
  l1 = tf.keras.layers.Dense(64, activation=None)(x)
  l1 = tf.keras.layers.Activation('sigmoid')(l1)
  y = tf.keras.layers.Dense(reduced_action_size, activation='linear')(l1)

  model = tf.keras.models.Model(inputs=[x], outputs=[y, y])
  opt = tf.keras.optimizers.RMSprop(lr=0.0001)
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
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value]
      sorted_idx  = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     7, 8]
    elif ego_correct_lane_gap > 0:
      valid = [7,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value]
      sorted_idx  = [7,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     8
                     ]
    else:
      valid = [8,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value]
      sorted_idx  = [8,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     7
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
                     7, 8
                     ]
    elif ego_correct_lane_gap > 0:
      valid = [7,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [7,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     8
                     ]
    else:
      valid = [8,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [8,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     7
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
                     7, 8]
    elif ego_correct_lane_gap > 0:
      valid = [7,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [7,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     8]
    elif ego_correct_lane_gap < 0:
      valid = [8,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [8,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     7]
  return (set(valid), sorted_idx)

cfg_validity = DQNCfg(name = "validity",
                      play=False,
                      version=None,
                      resume = False,
                      state_size=2,
                      action_size=reduced_action_size,
                      low_target=None,
                      high_target=None,
                      gamma=None,
                      gamma_inc=None,
                      gamma_max=None,
                      epsilon=0,
                      epsilon_dec=0,
                      epsilon_min=0,
                      threshold=None,
                      memory_size=None,
                      traj_end_pred=None,
                      replay_batch_size=None,
                      traj_end_ratio= None,
                      _build_model=None,
                      model_rst_prob_list = [],
                      tf_cfg=None,
                      reshape=reshape_validity,
                      _select_actions=select_actions_validity)

class returnTrue():
  def __init__(self):
    pass
  def __call__(self, x):
    return True

cfg_safety = DQNCfg(name = "safety",
                    play = False,
                    version = "current",
                    resume = False,
                    state_size = 5 + 12*NUM_VEH_CONSIDERED,
                    action_size = reduced_action_size,
                    low_target=-1,
                    high_target=0,
                    gamma = 0.9,
                    gamma_inc = 1e-5,
                    gamma_max = 0.9,
                    epsilon = 0.6,
                    epsilon_dec = 1e-5,
                    epsilon_min = 0.6,
                    threshold = -0.15,
                    memory_size = 3200,
                    traj_end_pred = returnTrue(),
                    replay_batch_size = 320,
                    traj_end_ratio= 0.0001,
                    _build_model = build_model_safety,
                    model_rst_prob_list = [],
                    tf_cfg = tf_cfg_safety,
                    reshape = reshape_safety)

cfg_regulation = DQNCfg(name = "regulation",
                        play = False,
                        version = "current",
                        resume = False,
                        state_size = 4 + 2*NUM_LANE_CONSIDERED + 7*NUM_VEH_CONSIDERED,
                        action_size = reduced_action_size,
                        low_target=-1,
                        high_target=0,
                        gamma = 0.90,
                        gamma_inc = 1e-5,
                        gamma_max = 0.95,
                        epsilon=0.8,
                        epsilon_dec=1e-5,
                        epsilon_min=0.8,
                        threshold = -0.2,
                        memory_size = 64000,
                        traj_end_pred = returnTrue(),
                        replay_batch_size = 640,
                        traj_end_ratio= 0.0001,
                        _build_model = build_model_regulation,
                        model_rst_prob_list = [],
                        tf_cfg = tf_cfg_regulation,
                        reshape = reshape_regulation)

cfg_speed_comfort = DQNCfg(name = "speed_comfort",
                           play = False,
                           version = None,
                           resume=False,
                           state_size = 2,
                           action_size = reduced_action_size,
                           low_target=None,
                           high_target=None,
                           gamma = None,
                           gamma_inc = None,
                           gamma_max = None,
                           epsilon=0,
                           epsilon_dec=0,
                           epsilon_min=0,
                           threshold = None,
                           memory_size = None,
                           traj_end_pred = None,
                           replay_batch_size = None,
                           traj_end_ratio= None,
                           _build_model = None,
                           model_rst_prob_list = [],
                           tf_cfg = None,
                           reshape = reshape_speed_comfort,
                           _select_actions=select_actions_speed_comfort)
