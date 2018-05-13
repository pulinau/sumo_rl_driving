#!python3
__author__ = "Changjian Li"

from collections import deque
import random
import time

class ReplayMemory():
  def __init__(self, end_pred, memory_size=None):
    self.traj_mem = deque(maxlen = memory_size)
    self.end_pred = end_pred

  def add_traj(self, traj):
    traj_seg = []
    for state, action, reward, next_state, done in traj[::-1]:
      if done or self.end_pred(reward):
        end_reward, end_state, end_done = reward, next_state, done
        if len(traj_seg) != 0:
          self.traj_mem.append(traj_seg)
        traj_seg = []
        i = 0
      traj_seg.append((state, action, end_reward, end_state, end_done, i))
      i += 1
    self.traj_mem.append(traj_seg)

  def sample_end(self, n):
    indices = random.sample(range(len(self.traj_mem)), min(n, len(self.traj_mem)))
    return [self.traj_mem[i][-1] for i in indices]

  def sample_traj(self, n):
    traj_samp = random.sample(self.traj_mem, min(n, len(self.traj_mem)))
    return [x for traj in traj_samp for x in traj]

  def __len__(self):
    return len(self.traj_mem)