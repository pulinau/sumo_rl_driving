import os
import random
import logging
import numpy as np
from copy import deepcopy

from .utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, model_dir):
    assert config.memory_size >= config.history_length + 1, "replay memory size must be larger than history length"
    self.model_dir = model_dir

    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.float32)
    self.states = np.empty([self.memory_size, config.state_length], dtype = np.float32)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.history_length = config.history_length
    self.state_length = config.state_length
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0

  def add(self, state, reward, action, terminal):
    assert state.shape == (self.state_length,)
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.states[self.current, ...] = state
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def _get_history(self, index):
    assert self.count > 0, "replay memory is empty"
    # round index to range(0, memory_size), allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      indexes = range(index - (self.history_length - 1), index + 1)
    else:
      # otherwise use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
    # There may be terminal states in the history
    terminal_indexes = np.where(terminals[indexes] == True)
    if len(terminal_indexes) != 0:
      return deepcopy(self.states[indexes, ...])
    else:
      # the states after terminal state in the history is set to 0 
      return np.where(np.array(indexes) <= terminal_indexes[0], deepcopy(self.states[indexes, ...]), np.zeros(history_length, state_length))

  def sample(self):
    # memory must be large enough to include poststate and prestate of history_length
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    self.prehistories = np.empty((self.batch_size, self.history_length, self.state_length), dtype = np.float32)
    self.posthistories = np.empty((self.batch_size, self.history_length, self.state_length), dtype = np.float32)
    for i in range(self.batch_size):
      # find random index
      if self.count <= self.memory_size:
        # if replay memory is not full, wrap-over cannot happen
        index = random.randint(self.history_length, self.count - 1)
      else:
        # otherwise if replay memory is full, sample from a range where wrap-over is avoided
        start_index = (self.current + self.history_length) % self.memory_size
        if start_index > self.current:
          start_index -= self.memory_size
        index = random.randint(start_index, self.current - 1)
      indexes.append(index)
      prehistories[i, ...] = self._get_history(index - 1)
      posthistoies[i, ...] = self._get_history(index)
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return deepcopy(prehistories), deepcopy(actions), deepcopy(rewards), deepcopy(posthistoies), deepcopy(terminals)
