#!python3
__author__ = "Changjian Li"

from collections import deque
import random
import time

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
    #print("sampling_end" ,time.time())
    indices = random.sample(range(len(self.traj_mem)), min(n, len(self.traj_mem)))
    return [self.traj_mem[i][-1] for i in indices]

  def sample_traj(self, n):
    #print("sampling_traj", time.time())
    traj_samp = random.sample(self.traj_mem, min(n, len(self.traj_mem)))
    return [x for traj in traj_samp for x in traj]

  def __len__(self):
    return len(self.traj_mem)