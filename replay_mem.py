#!python3
__author__ = "Changjian Li"

from collections import deque
import random
import time

class ReplayMemory():
  def __init__(self, memory_size, end_pred):
    self.traj_mem = deque(maxlen = memory_size)
    self.end_pred = end_pred

  def add_traj(self, traj):
    traj = []
    for state, action, reward, next_state, done in traj[::-1]:
      if done or self.end_pred(reward):
        end_reward, end_state, end_done = reward, next_state, done
        if len(traj) != 0:
          self.traj_mem.append(traj)
        traj = []
        i = 0
      traj.append((state, action, end_reward, end_state, end_done, i))
      i += 1
    self.traj_mem.append(traj)

  def sample_end(self, n):
    indices = random.sample(range(len(self.traj_mem)), min(n, len(self.traj_mem)))
    return [self.traj_mem[i][-1] for i in indices]

  def sample_traj(self, n):
    traj_samp = random.sample(self.traj_mem, min(n, len(self.traj_mem)))
    return [x for traj in traj_samp for x in traj]

  def __len__(self):
    return len(self.traj_mem)