#!python3
__author__ = "Changjian Li"

from collections import deque
import random
import dill as pickle
import time
import multiprocessing as mp
from multiprocessing.managers import BaseManager

class ReplayMemory():
  """
  Replay memory for dynamic n-step look-ahead. Since the problem domain has very sparse reward,
  instead of using a fixed look-ahead time step, bootstrap is only done when the reward is actually received.
  Insignificant reward in between can be ignored by supplying a predicate, which is treated as 0 reward.
  """
  def __init__(self, max_len):
    """
    :param max_len: max replay memory size
    """
    assert max_len > 0, "max_len must be greater than zero"
    self.max_len = max_len

    self.lock = mp.Lock()
    self.lock.acquire()

    self.states = []
    self.actions = []
    self.rewards = []
    self.next_states = []
    self.not_dones = []
    self.steps = []

    self.end_states = []
    self.end_actions = []
    self.end_rewards = []
    self.end_next_states = []
    self.end_not_dones = []
    self.end_steps = []

    self.lock.release()

    self.avg_traj_seg_len = 0

  def add_traj(self, traj, end_pred):
    """
    :param traj: list of state transitions
    :param end_pred: decide whether the reward is significant enough to be considered
    :return:
    """
    self.lock.acquire()

    traj_seg = []
    for i, (state, action, reward, next_state, done) in enumerate(traj[::-1]):
      if i == 0 or done or end_pred(reward):
        end_reward, end_state, end_done = reward, next_state, done
        #if not done and i == 0:
        #  end_done = True
        if len(traj_seg) != 0:
          self.avg_traj_seg_len = (len(self.end_actions) * self.avg_traj_seg_len + len(traj_seg)) / \
                                  (len(self.end_actions) + 1)
          for j, x in enumerate(traj_seg):
            self._add(x, j == len(traj_seg)-1)
        traj_seg = []
        step = 0
      traj_seg.append((state, action, end_reward, end_state, end_done, step))
      step += 1
    self.avg_traj_seg_len = (len(self.end_actions) * self.avg_traj_seg_len + len(traj_seg)) / \
                            (len(self.end_actions) + 1)
    for j, x in enumerate(traj_seg):
      self._add(x, j == len(traj_seg)-1)

    self.lock.release()

  def _add(self, tran, is_end):
    state, action, reward, next_state, done, step = tran

    if len(self.actions) > 2 * self.max_len:
      for i in range(len(self.states)):
        self.states[i] = self.states[i][self.max_len:]
        self.next_states[i] = self.next_states[i][self.max_len:]
      self.actions = self.actions[self.max_len:]
      self.rewards = self.rewards[self.max_len:]
      self.not_dones = self.not_dones[self.max_len:]
      self.steps = self.steps[self.max_len:]

    self.actions += [action]
    self.rewards += [reward]
    self.not_dones += [not done]
    self.steps += [step]
    if len(self.states) == 0:
      for i in range(len(state)):
        self.states += [state[i]]
        self.next_states += [next_state[i]]
    else:
      for i in range(len(self.states)):
        self.states[i] += state[i]
        self.next_states[i] += next_state[i]

    if is_end:
      cap = self.max_len//(self.avg_traj_seg_len+1) + 1
      if len(self.end_actions) > 2 * cap:
        for i in range(len(self.end_states)):
          self.end_states[i] = self.end_states[i][cap:]
          self.end_next_states[i] = self.end_next_states[i][cap:]
        self.end_actions = self.end_actions[cap:]
        self.end_rewards = self.end_rewards[cap:]
        self.end_not_dones = self.end_not_dones[cap:]
        self.end_steps = self.end_steps[cap:]

      self.end_actions += [action]
      self.end_rewards += [reward]
      self.end_not_dones += [not done]
      self.end_steps += [step]
      if len(self.end_states) == 0:
        for i in range(len(state)):
          self.end_states += [state[i]]
          self.end_next_states += [next_state[i]]
      else:
        for i in range(len(self.end_states)):
          self.end_states[i] += state[i]
          self.end_next_states[i] += next_state[i]

  def _sample_end(self, n):
    self.lock.acquire()

    assert n > 0, "sample size must be positive"
    assert len(self.end_actions) > 0, "replay memory empty"
    indices = random.sample(range(len(self.end_actions)), min(n, len(self.end_actions)))
    actions = [self.end_actions[i] for i in indices]
    rewards = [self.end_rewards[i] for i in indices]
    not_dones = [self.end_not_dones[i] for i in indices]
    steps = [self.end_steps[i] for i in indices]
    states = [[x[i] for i in indices]
              for x in self.end_states]
    next_states = [[x[i] for i in indices]
                   for x in self.end_next_states]
    samp = (states, actions, rewards, next_states, not_dones, steps)

    self.lock.release()
    return samp

  def _sample_traj(self, n):
    self.lock.acquire()

    assert n > 0, "sample size must be positive"
    assert len(self.actions) > 0, "replay memory empty"
    indices = random.sample(range(len(self.actions)), min(n, len(self.actions)))
    actions = [self.actions[i] for i in indices]
    rewards = [self.rewards[i] for i in indices]
    not_dones = [self.not_dones[i] for i in indices]
    steps = [self.steps[i] for i in indices]
    states = [[x[i] for i in indices]
              for x in self.states]
    next_states = [[x[i] for i in indices]
                   for x in self.next_states]
    samp = (states, actions, rewards, next_states, not_dones, steps)

    self.lock.release()
    return samp

  def sample(self, n, traj_end_ratio):
    assert traj_end_ratio < 1 and traj_end_ratio > 0, "traj_end_ratio must lie between 0 and 1"
    alpha = self.avg_traj_seg_len / (1/traj_end_ratio -1 + self.avg_traj_seg_len)
    end_states, end_actions, end_rewards, \
      end_next_states, end_not_dones, end_steps =  self._sample_end(max(int(alpha*n), 1))
    states, actions, rewards, \
      next_states, not_dones, steps = self._sample_traj(max(int((1-alpha)*n), 1))
    actions += end_actions
    rewards += end_rewards
    not_dones += end_not_dones
    steps += end_steps
    for i in range(len(states)):
      states[i] += end_states[i]
      next_states[i] += end_next_states[i]
    return (states, actions, rewards, next_states, not_dones, steps)

  def size(self):
    self.lock.acquire()
    l = len(self.actions)
    self.lock.release()
    return l

class ReplayMemoryManager(BaseManager):
  pass
ReplayMemoryManager.register('ReplayMemory', ReplayMemory)