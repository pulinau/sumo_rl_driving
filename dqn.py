#!python3
__author__ = "Changjian Li"

from include import *
from sumo_cfgs import *
from utils import class_vars
import random
import tensorflow as tf
import numpy as np
from replay_mem import ReplayMemory
from action import loosen_correct_actions
import time

class DQNCfg():
  def __init__(self, 
               name, 
               play, 
               state_size, 
               action_size,
               pretrain_low_target,
               pretrain_high_target,
               gamma,
               gamma_inc,
               gamma_max,
               epsilon,
               epsilon_dec,
               epsilon_min,
               threshold,
               memory_size,
               traj_end_pred,
               replay_batch_size,
               _build_model,
               tf_cfg,
               reshape,
               _select_actions = None):
    self.name = name
    self.play = play # whether it's training or playing
    self.state_size = state_size
    self.action_size = action_size
    self.pretrain_low_target = pretrain_low_target
    self.pretrain_high_target = pretrain_high_target
    self.gamma = gamma
    self.gamma_inc = gamma_inc
    self.gamma_max = gamma_max
    self.epsilon = epsilon
    self.epsilon_dec = epsilon_dec
    self.epsilon_min = epsilon_min
    self.threshold = threshold
    self.memory_size = memory_size
    self.traj_end_pred = traj_end_pred
    self.replay_batch_size = replay_batch_size
    self._build_model = _build_model
    self.tf_cfg = tf_cfg
    self.reshape = reshape
    self._select_actions = _select_actions

class DQNAgent:
  def __init__(self, sumo_cfg, dqn_cfg):
    _attrs = class_vars(dqn_cfg)
    for _attr in _attrs:
      setattr(self, _attr, getattr(dqn_cfg, _attr))

    if self._select_actions is not None:
      return

    assert self.memory_size >= self.replay_batch_size

    tf.keras.backend.set_session(tf.Session(config=self.tf_cfg))
    if self.play == True:
      self.epsilon = 0
      self.model = self._load(sumo_cfg)
    else:
      self.memory = ReplayMemory(self.traj_end_pred, self.memory_size)
      self.model = self._build_model()
      self.target_model = self._build_model()

  def remember(self, traj):
    if self.play == True or self._select_actions is not None:
      return
    traj = [(self.reshape(obs_dict), action, reward, self.reshape(next_obs_dict), done)
            for obs_dict, action, reward, next_obs_dict, done in traj]
    self.memory.add_traj(traj)

  def select_actions(self, obs_dict):
    if self._select_actions is not None:
      return self._select_actions(obs_dict)

    act_values = self.model.predict(self.reshape(obs_dict))[0]

    sorted_idx = np.argsort(act_values)

    action_set = set(np.where(act_values > self.threshold)[0]) | set(sorted_idx[-6:])
    explore_set = set([action for action in range(self.action_size) if np.random.rand() <= self.epsilon])
    explore_set = explore_set - action_set

    return (action_set, explore_set, sorted_idx)

  def pretrain(self, traj_list, ep):
    if self.play == True or len(traj_list) == 0  or self._select_actions is not None:
      return

    self.pretrain_mem = ReplayMemory(end_pred=True)
    for traj in traj_list:
      traj = [(self.reshape(obs_dict), action, reward, None, done)
              for obs_dict, action, reward, next_obs_dict, done in traj]
      self.pretrain_mem.add_traj(traj)

    states = [s[0][0] for s in self.pretrain_mem.traj_mem]
    actions = [s[0][1] for s in self.pretrain_mem.traj_mem]

    state_idx, correct_actions = loosen_correct_actions(actions)

    temp = []
    for i in range(len(states[0])):
      arr = np.reshape(np.array([], dtype=np.float32), (0,) + states[0][i][0].shape)
      for x in states:
        arr = np.append(arr, x[i], axis=0)
      temp += [arr]
    states = temp

    targets_f = self.pretrain_low_target * np.ones((states[0].shape[0], self.action_size))
    targets_f[state_idx, correct_actions] = self.pretrain_high_target

    print(self.name, " pretrain start")
    self.model.fit(states, targets_f, epochs = ep)
    self.pretrain_mem = None
    print(self.name, " pretrain complete")

  def replay(self):
    if self.play == True or len(self.memory) == 0 or self._select_actions is not None:
      return

    #print(self.name, " sampling starts", time.time())
    minibatch = self.memory.sample_end(self.replay_batch_size) + self.memory.sample_traj(self.replay_batch_size//8)
    #print(self.name, " sampling ends", time.time())

    states = [s[0] for s in minibatch]
    actions = [s[1] for s in minibatch]
    rewards = [s[2] for s in minibatch]
    next_states = [s[3] for s in minibatch]
    dones = [np.float(not s[4]) for s in minibatch]
    steps = [s[5] for s in minibatch]
    
    actions = np.array(actions)
    rewards = np.array(rewards)
    steps = np.array(steps)
    temp = []

    for i in range(len(states[0])):
      arr = np.reshape(np.array([], dtype=np.float32), (0,) + states[0][i][0].shape)
      for x in states:
        arr = np.append(arr, x[i], axis = 0)
      temp += [arr]
    states = temp
    temp = []
    for i in range(len(next_states[0])):
      arr = np.reshape(np.array([], dtype = np.float32), (0, ) + next_states[0][i][0].shape)
      for x in next_states:
        arr = np.append(arr, x[i], axis = 0)
      temp += [arr]
    next_states = temp

    backup = (self.gamma**(steps+1)) * np.array(dones) * np.amax(self.target_model.predict_on_batch(next_states), axis = 1)
    # clamp targets larger than zero to zero

    backup[np.where(backup > 0)] = 0
    targets = (self.gamma**steps)*rewards + backup
    targets_f = self.target_model.predict_on_batch(states)
    targets_f[np.arange(targets_f.shape[0]), actions] = targets


    print(self.name, targets_f[0])
    #print(self.name , " training starts", time.time(), flush = True)
    id = random.randint(0, 65536)
    for i in range(2):
      print(self.name, " id:", id, " ep:", i, self.model.train_on_batch(states, targets_f))

    #print(self.name, " training ends", time.time(), flush = True)
    if np.any(np.isnan(self.model.predict_on_batch(states))):
      print("\n###################NAN...####################\n")
      import time
      while True:
        time.sleep(1000)

    if self.gamma < self.gamma_max:
      self.gamma += self.gamma_inc
    if self.epsilon > self.epsilon_min:
      self.epsilon -= self.epsilon_dec

  def update_target(self):
    if self.play == True or self._select_actions is not None:
      return
    self.target_model.set_weights(self.model.get_weights())

  def _load(self, sumo_cfg):
    return tf.keras.models.load_model(self.name + ".sav", custom_objects={"tf": tf, "NUM_VEH_CONSIDERED": sumo_cfg.NUM_VEH_CONSIDERED})

  def save(self):
    if self.play == True or self._select_actions is not None:
      return
    self.model.save(self.name + ".sav")
