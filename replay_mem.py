#!python3
__author__ = "Changjian Li"

from collections import deque
import random
import time

class ReplayMemory():
  def __init__(self, end_pred, memory_size=None):
    self.traj_mem = deque(maxlen = memory_size)
    self.end_pred = end_pred
    self.avg_traj_seg_len = 0

  def add_traj(self, traj):
    traj_seg = []
    for i, (state, action, reward, next_state, done) in enumerate(traj[::-1]):
      if i == 0 or done or self.end_pred(reward):
        end_reward, end_state, end_done = reward, next_state, done
        #if not done and i == 0:
        #  end_done = True
        if len(traj_seg) != 0:
          self.avg_traj_seg_len = (len(traj_mem) * self.avg_traj_seg_len + len(traj_seg)) / (len(traj_mem) + 1)
          self.traj_mem.append(traj_seg)
        traj_seg = []
        step = 0
      traj_seg.append((state, action, end_reward, end_state, end_done, step))
      step += 1
    self.avg_traj_seg_len = (len(traj_mem) * self.avg_traj_seg_len + len(traj_seg)) / (len(traj_mem) + 1)
    self.traj_mem.append(traj_seg)

  def sample_end(self, n):
    indices = random.sample(range(len(self.traj_mem)), min(n, len(self.traj_mem)))
    return [self.traj_mem[i][-1] for i in indices]

  def sample_traj(self, n):
    traj_samp = random.sample(self.traj_mem, min(n, len(self.traj_mem)))
    return [x for traj in traj_samp for x in traj]

  def sample(self, n, traj_end_ratio):
    assert traj_end_ratio < 1 and traj_end_ratio > 0, "traj_end_ratio must lie between 0 and 1"
    alpha = self.avg_traj_seg_len / (1/traj_end_ratio -1 + self.avg_traj_seg_len)
    return self.sample_end(int(alpha*n)) + self.sample_traj(int((1-alpha)*n))

  def __len__(self):
    return len(self.traj_mem)