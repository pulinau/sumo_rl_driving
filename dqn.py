#!python3
from include import *
from utils import class_vars
import random
import gym
import numpy as np
from collections import deque
import inspect

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
    self.reshape = reshape

class DQNAgent:
  def __init__(self, sumo_cfg, dqn_cfg):
    _attrs = class_vars(dqn_cfg)
    for _attr in _attrs:
      setattr(self, _attr, getattr(dqn_cfg, _attr))

    assert self.memory_size >= self.replay_batch_size

    if self.play == True:
      self.epsilon = 0
      self.model = self.load(sumo_cfg)
    else:
      self.memory = deque(maxlen = self.memory_size)
      self.model = self._build_model()

  def remember(self, state, action, reward, next_state, env_state):
    self.memory.append((state, action, reward, next_state, env_state))

  def select_actions(self, state):
    act_values = self.model.predict(state)[0]

    action_set = set(np.where(act_values > self.threshold)[0])
    explore_set = set([action for action in range(self.action_size) if np.random.rand() <= self.epsilon])

    return action_set, explore_set

  def learn(self, state, action, reward, next_state, env_state):
    if self.play == True:
      return
    target = reward
    if env_state == EnvState.NORMAL:
      target = (reward + self.gamma *
                np.amax(self.model.predict(next_state)[0]))
    target_f = self.model.predict(state)
    target_f[0][action] = target
    self.model.fit(state, target_f, epochs=1, verbose=0)
  
  def replay(self):
    if self.play == True:
      return
    if len(self.memory) < self.replay_batch_size:
      batch_size = len(self.memory)
    minibatch = np.array(random.sample(self.memory, self.replay_batch_size))
    states = [s[0] for s in minibatch]
    actions = [s[1] for s in minibatch]
    rewards = [s[2] for s in minibatch]
    next_states = [s[3] for s in minibatch]
    env_states = [np.float(s[4] == EnvState.NORMAL) for s in minibatch]
    
    actions = np.array(actions)
    rewards = np.array(rewards)
    temp = []
    for i in range(len(states[0])):
      temp += [np.array([x[i][0] for x in states])]
    states = temp
    temp = []
    for i in range(len(next_states[0])):
      temp += [np.array([x[i][0] for x in next_states])]
    next_states = temp      

    print(self.model)
    targets = rewards + self.gamma * np.array(env_states) * np.amax(self.model.predict(next_states), axis = 1)
    targets_f = self.model.predict(states)
    targets_f[np.arange(targets_f.shape[0]), actions] = targets
    self.model.fit(states, targets_f, epochs=3, verbose=0)

    if self.gamma < self.gamma_max:
      self.gamma += self.gamma_inc

  def load(self, sumo_cfg):
    import tensorflow as tf
    return tf.keras.models.load_model(self.name + ".sav", custom_objects={"tf": tf, })#"NUM_VEH_CONSIDERED": sumo_cfg.NUM_VEH_CONSIDERED})

  def save(self):
    self.model.save(self.name + ".sav")
