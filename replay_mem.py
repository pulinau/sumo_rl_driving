#!python3
__author__ = "Changjian Li"

from collections import deque
import random

class ReplayMemory():
  def __init__(self, memory_size, end_pred):
    self.traj_mem = deque(maxlen = memory_size)
    self.end_pred = end_pred
    self.traj = []

  def append(self, state, action, reward, next_state, done):
    self.traj += [(state, action, reward, next_state, done)]
    if done or self.end_pred(reward):
      self.traj = [(x[0], x[1], reward, next_state, done, i) for i, x in enumerate(self.traj[::-1])]
      self.traj_mem.append(self.traj)
      self.traj = []

  def sample_end(self, n):
    traj_mem = list(self.traj_mem)
    indices = random.sample(range(len(traj_mem)), min(n, len(traj_mem)))
    return [traj_mem[i][-1] for i in indices]

  def sample_traj(self, n):
    traj_mem = list(self.traj_mem)
    traj_samp = random.sample(traj_mem, min(n, len(traj_mem)))
    return [x for traj in traj_samp for x in traj]

  def __len__(self):
    return len(self.traj_mem)