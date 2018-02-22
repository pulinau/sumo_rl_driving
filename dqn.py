#!python3
from include import *
from utils import class_vars
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import inspect

class DQNCfg():
  def __init__(self, 
               name, 
               state_size, 
               action_size, 
               gamma, 
               epsilon,
               threshold,
               memory_size, 
               _build_model, 
               reshape):
    self.name = name
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = gamma
    self.epsilon = epsilon
    self.threshold = threshold
    self.memory_size = memory_size
    self._build_model = _build_model
    self.reshape = reshape

class DQNAgent:
  def __init__(self, sumo_cfg, dqn_cfg):
    _attrs = class_vars(dqn_cfg)
    for _attr in _attrs:
      setattr(self, _attr, getattr(dqn_cfg, _attr))
    
    self.memory = deque(maxlen = self.memory_size)
    self.model = self._build_model(sumo_cfg, dqn_cfg)

  def remember(self, state, action, reward, next_state, env_state):
    self.memory.append((state, action, reward, next_state, env_state))

  def get_action_set(self, state, in_action_set):
    act_values = self.model.predict(state)[0]
    if np.all(act_values < self.threshold):
      print("no available action for" + self.name)
      return in_action_set
    out_action_set = set()
    for action in in_action_set:
      if np.random.rand() <= self.epsilon:
        out_action_set = out_action_set or {action}
    out_action_set = out_action_set or (set(np.where(act_values > self.threshold)[0]) and in_action_set)
    return out_action_set
  
  def learn(self, state, action, reward, next_state, env_state):
    target = reward
    if env_state == EnvState.NORMAL:
      target = (reward + self.gamma *
                np.amax(self.model.predict(next_state)[0]))
    target_f = self.model.predict(state)
    target_f[0][action] = target
    self.model.fit(state, target_f, epochs=1, verbose=0)
  
  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, env_state in minibatch:
      target = reward
      if env_state == EnvState.NORMAL:
        target = (reward + self.gamma *
                  np.amax(self.model.predict(next_state)[0]))
      target_f = self.model.predict(state)
      target_f[0][action] = target
      self.model.fit(state, target_f, epochs=1, verbose=0)

  def load(self):
    self.model.load_weights(self.name + ".sav")

  def save(self):
    self.model.save_weights(self.name + ".sav")
