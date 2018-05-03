#!python3
__author__ = "Changjian Li"

from include import *
from sumo_cfgs import *
from utils import class_vars
import random
import tensorflow as tf
import numpy as np
from replay_mem import ReplayMemory
import time

class DQNCfg():
  def __init__(self, 
               name, 
               play, 
               state_size, 
               action_size, 
               gamma,
               gamma_inc,
               gamma_max,
               epsilon,
               threshold,
               memory_size,
               replay_batch_size,
               _build_model,
               tf_cfg,
               reshape):
    self.name = name
    self.play = play # whether it's training or playing
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = gamma
    self.gamma_inc = gamma_inc
    self.gamma_max = gamma_max
    self.epsilon = epsilon
    self.threshold = threshold
    self.memory_size = memory_size
    self.replay_batch_size = replay_batch_size
    self._build_model = _build_model
    self.tf_cfg = tf_cfg
    self.reshape = reshape

class DQNAgent:
  def __init__(self, sumo_cfg, dqn_cfg):
    _attrs = class_vars(dqn_cfg)
    for _attr in _attrs:
      setattr(self, _attr, getattr(dqn_cfg, _attr))

    assert self.memory_size >= self.replay_batch_size

    tf.keras.backend.set_session(tf.Session(config=self.tf_cfg))
    if self.play == True:
      self.epsilon = 0
      self.model = self.load(sumo_cfg)
    else:
      self.memory = ReplayMemory(self.memory_size, lambda x: x < -0.1)
      self.model = self._build_model()
      self.target_model = self._build_model()

  def remember(self, traj):
    if self.play == True:
      return
    traj = [(self.reshape(obs_dict), action, reward, self.reshape(next_obs_dict), done)
            for obs_dict, action, reward, next_obs_dict, done in traj]
    self.memory.add_traj(traj)

  def select_actions(self, obs_dict):
    act_values = self.model.predict(self.reshape(obs_dict))[0]

    idx = np.argsort(act_values)
    if len(set(np.where(act_values > self.threshold)[0])) == 0:
      self.threshold -= 0.001
    if len(set(np.where(act_values > self.threshold)[0])) > 6:
      self.threshold += 0.001
    action_set = set(np.where(act_values > self.threshold)[0]) | set(act_values[idx[-6:]])
    explore_set = set([action for action in range(self.action_size) if np.random.rand() <= self.epsilon])
    explore_set = explore_set - action_set

    return (action_set, explore_set)
  
  def replay(self):
    if self.play == True or len(self.memory) == 0:
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

    #print(self.name , " training starts", time.time(), flush = True)
    self.model.train_on_batch(states, targets_f)
    #print(self.name, " training ends", time.time(), flush = True)
    if np.any(np.isnan(self.model.predict_on_batch(states))):
      print("\nNAN...")
      import time
      while True:
        time.sleep(100)

    if self.gamma < self.gamma_max:
      self.gamma += self.gamma_inc

  def update_target(self):
    if self.play == True:
      return
    self.target_model.set_weights(self.model.get_weights())

  def load(self, sumo_cfg):
    return tf.keras.models.load_model(self.name + ".sav", custom_objects={"tf": tf, "NUM_VEH_CONSIDERED": sumo_cfg.NUM_VEH_CONSIDERED})

  def save(self):
    if self.play == True:
      return
    self.model.save(self.name + ".sav")
