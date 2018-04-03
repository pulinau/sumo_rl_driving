#!python3
__author__ = "Changjian Li"

import argparse

from action import *
from observation import *
from sumo_gym import *

from dqn import *
import random
from multiprocessing import Process
import multiprocessing

# --------------------------
#          SUMO
# --------------------------
SUMO_TOOLS_DIR = "/home/ken/project/sumo-bin/tools"
SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo"
SUMO_CONFIG = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.sumocfg"
SUMO_TIME_STEP = 0.1

# "--net-file" and "route_file"
SUMO_CMD = [SUMO_BIN,
            "-c", "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.sumocfg",
            #"--no-warnings", "true", 
            "--time-to-teleport", "-1", 
            "--collision.action", "none", 
            "--collision.check-junctions", "true", 
            "--step-length", str(SUMO_TIME_STEP)]
#            "--lanechange.duration", "2"]
NET_XML_FILE = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.net.xml"
ROU_XML_FILE_LIST = ["test0.rou.xml"]

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
MAX_COMFORT_ACCEL_LEVEL = ActionAccel.MINACCEL
MAX_COMFORT_DECEL_LEVEL = ActionAccel.MINDECEL

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

def build_model_safety():
  import tensorflow as tf
  ego_input = tf.keras.layers.Input(shape=(4, ))
  env_input = tf.keras.layers.Input(shape=(10* NUM_VEH_CONSIDERED, 1, 1))
  l1_0 = tf.keras.layers.Dense(64, activation = None)(ego_input)
  l1_1 = tf.keras.layers.Conv2D(64, kernel_size = (10, 1),
                                strides = (10, 1), padding = 'valid', 
                                activation = None)(env_input)
  l1_1 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.reshape(x, [-1, NUM_VEH_CONSIDERED, 64]), axis=1))(l1_1)
  l1 = tf.keras.layers.add([l1_0, l1_1])
  l1 = tf.keras.layers.Activation(activation="relu")(l1)
  l2 = tf.keras.layers.Dense(64)(l1)
  l2 = tf.keras.layers.BatchNormalization()(l2)
  l2 = tf.keras.layers.Activation('sigmoid')(l2)
  l3 = tf.keras.layers.Dense(64)(l2)
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

def build_model_regulation():
  import tensorflow as tf
  ego_input = tf.keras.layers.Input(shape=(6 + 2*NUM_LANE_CONSIDERED, ))
  env_input = tf.keras.layers.Input(shape=(16*NUM_VEH_CONSIDERED, 1, 1))
  l1_0 = tf.keras.layers.Dense(64, activation = None)(ego_input)
  l1_1 = tf.keras.layers.Conv2D(64, kernel_size = (16, 1), strides = (16, 1), padding = 'valid',
                activation = None)(env_input)
  l1_1 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.reshape(x, [-1, NUM_VEH_CONSIDERED, 64]), axis=1))(l1_1)
  l1 = tf.keras.layers.add([l1_0, l1_1])
  l1 = tf.keras.layers.Activation(activation="relu")(l1)
  l2 = tf.keras.layers.Dense(64)(l1)
  l2 = tf.keras.layers.BatchNormalization()(l2)
  l2 = tf.keras.layers.Activation('sigmoid')(l2)
  l3 = tf.keras.layers.Dense(64)(l2)
  l3 = tf.keras.layers.BatchNormalization()(l3)
  l3 = tf.keras.layers.Activation('sigmoid')(l3)
  y = tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear')(l3)
  model = tf.keras.models.Model(inputs = [ego_input, env_input], outputs=y)
  model.compile(loss='mse', optimizer="SGD")
  return model

def reshape_comfort(obs_dict):
  return np.reshape(np.array([0], dtype = np.float32), (1, -1)) 

def build_model_comfort():
  import tensorflow as tf
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(8, input_dim=1, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
  model.add(tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear'))
  model.compile(loss='mse', optimizer="SGD")
  return model

def reshape_speed(obs_dict):
  return np.reshape(np.array(obs_dict["ego_speed"]/MAX_VEH_SPEED, dtype = np.float32), (1, -1))

def build_model_speed():
  import tensorflow as tf
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
               MAX_COMFORT_ACCEL_LEVEL, 
               MAX_COMFORT_DECEL_LEVEL)

def select_action(action_set_list, explr_set_list, dqn_cfg_list):
  valid = set()
  invalid = [] # invalid holds the exploration actions

  for action_set, explr_set, name in zip(action_set_list, explr_set_list, [dqn_cfg.name for dqn_cfg in dqn_cfg_list]):
    if len(valid) == 0:
      valid = valid or action_set
      invalid += [(x, "exploration: " + name) for x in explr_set]
    else:
      invalid += [(x, "exploration: " + name) for x in (explr_set and valid)]
    valid = valid and action_set
    if len(valid) == 0:
      print("no available action for " + name)
      break

  valid = [(x, "valid") for x in valid]

  if len(valid) == 0 and len(invalid) == 0:
    return (random.sample(range(action_size), 1)[0], "exploration")
  else:
    return random.sample(valid + invalid, 1)[0]

def Qlearning(conn, sumo_cfg, dqn_cfg):
  agt = DQNAgent(sumo_cfg, dqn_cfg)

  while conn.recv() == True:
    obs_dict = conn.recv()
    state = agt.reshape(obs_dict)

    action_set, explr_set = agt.select_actions(state)
    conn.send((action_set, explr_set))

    next_obs_dict, reward, env_state, action = conn.recv()
    next_state = agt.reshape(next_obs_dict)
    agt.remember()

    agt.replay()

    if conn.recv() == True:
      agt.save()

  conn.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--play")
  args = parser.parse_args()

  env = MultiObjSumoEnv(sumo_cfg)
  env_state = EnvState.NORMAL
  EPISODES = 10000
  if args.play:
    print("True")
    for dqn_cfg in [cfg_safety, cfg_regulation, cfg_comfort, cfg_speed]:
      dqn_cfg.play = True
    EPISODES = 10

  dqn_cfg_list = [cfg_safety, cfg_regulation, cfg_comfort, cfg_speed]
  parent_conn_list, child_conn_list = zip(*[mp.Pipe() for _ in range(4)])

  p_list = [mp.Process(target=Qlearning, args=(conn, sumo_cfg, dqn_cfg)) for conn, dqn_cfg in
            zip(child_conn_list, dqn_cfg_list)]
  [p.start() for p in p_list]

  for e in range(EPISODES):
    print("episode: {}/{}".format(e, EPISODES))
    obs_dict = env.reset()

    for step in range(6400):
      # env.agt_ctrl = False
      if step == 0:
        if random.uniform(0, 1) < 0.5:
          env.agt_ctrl = True
        else:
          env.agt_ctrl = False
      else:
        if random.uniform(0, 1) < 0.05:
          if env.agt_ctrl == True:
            env.agt_ctrl = False
          else:
            env.agt_ctrl = True

      [conn.send(True) for conn in parent_conn_list]

      # send obs_dict
      [conn.send(obs_dict) for conn in parent_conn_list]

      # select action
      action_set_list, explr_set_list = zip(*[conn.recv() for conn in parent_conn_list])
      action, action_info = select_action(action_set_list, explr_set_list)
      if env.agt_ctrl == False:
        action_info == "sumo"

      # act
      next_obs_dict, reward_list, env_state, action_dict = env.step(
        {"lane_change": ActionLaneChange(action // 7), "accel_level": ActionAccel(action % 7)})
      if env_state == EnvState.DONE:
        print("Ego successfully drived out of scene, step: ", step)
        break
      action = action_dict["lane_change"].value * 7 + action_dict["accel_level"].value
      [conn.send((obs_dict, reward, env_state, action)) for conn, reward in zip(parent_conn_list, reward_list)]

      # save model
      if e % 100 == 1:
        [conn.send(True) for conn in parent_conn_list]
      else:
        [conn.send(False) for conn in parent_conn_list]

      obs_dict = next_obs_dict
      if env_state != EnvState.NORMAL or step == 6400 - 1:
        print("Simulation Terminated, step: ", step, action_dict, action_info, reward_list, env_state, env.agt_ctrl)
        break

  [conn.send(False) for conn in parent_conn_list]
  [p.join() for p in p_list]