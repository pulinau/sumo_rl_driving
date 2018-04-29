#!python3
__author__ = "Changjian Li"

from collections import deque
import random
import time

class ReplayMemory():
  def __init__(self, memory_size, end_pred):
    self.traj_mem = deque(maxlen = memory_size)

  def add_traj(self, traj):
    state, action, reward, next_state, done = traj[-1]
    self.traj = [(x[0], x[1], reward, next_state, done, i) for i, x in enumerate(self.traj[::-1])]
    self.traj_mem.append(self.traj)

  def sample_end(self, n):
    indices = random.sample(range(len(self.traj_mem)), min(n, len(self.traj_mem)))
    return [self.traj_mem[i][-1] for i in indices]

  def sample_traj(self, n):
    traj_samp = random.sample(self.traj_mem, min(n, len(self.traj_mem)))
    return [x for traj in traj_samp for x in traj]

  def __len__(self):
    return len(self.traj_mem)