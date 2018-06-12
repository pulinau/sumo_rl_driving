#!python3
__author__ = "Changjian Li"

from collections import deque
import random
import dill as pickle
import time
import multiprocessing as mp

class ReplayMemory():
  """
  Replay memory for dynamic n-step look-ahead. Since the problem domain has very sparse reward,
  instead of using a fixed look-ahead time step, bootstrap is only done when the reward is actually received.
  Insignificant reward in between can be ignored by supplying a predicate, which is treated as 0 reward.
  """
  def __init__(self, max_len):
    """
    :param end_pred: decide whether the reward is significant enough to be considered
    :param max_len: max replay momory size
    """
    assert max_len > 0

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

    self.avg_traj_seg_len = 0

  def add_traj(self, traj, end_pred):
    traj_seg = []
    for i, (state, action, reward, next_state, done) in enumerate(traj[::-1]):
      if i == 0 or done or end_pred(reward):
        end_reward, end_state, end_done = reward, next_state, done
        #if not done and i == 0:
        #  end_done = True
        if len(traj_seg) != 0:
          self.avg_traj_seg_len = (len(self.traj_mem) * self.avg_traj_seg_len + len(traj_seg)) / \
                                  (len(self.traj_mem) + 1)
          for i, x in enumerate(traj_seg):
            self.add(x, i == len(traj_seg))
        traj_seg = []
        step = 0
      traj_seg.append((state, action, end_reward, end_state, end_done, step))
      step += 1
    self.avg_traj_seg_len = (len(self.end_actions) * self.avg_traj_seg_len + len(traj_seg)) / \
                            (len(self.end_actions) + 1)
    for i, x in enumerate(traj_seg):
      self.add(x, i == len(traj_seg))

  def add(self, tran, is_end):
    state, action, reward, next_state, done, step = tran

    if len(self.actions) > 2 * max_len:
      for i in range(len(self.states)):
        self.states[i] = self.states[i][max_len:]
      self.actions = self.actions[max_len:]
      self.rewards = self.rewards[man_len:]
      for i in range(len(self.next_states)):
        self.next_states[i] = self.next_states[i][max_len:]
      self.not_dones = self.not_dones[max_len:]
      self.steps = self.steps[max_len:]

    self.actions += [action]
    self.rewards += [reward]
    self.not_dones += [not done]
    self.steps += [step]
    if len(self.states) == 0:
      for i in range(len(state)):
        self.states += [state[i]]
    else:
      for i in range(len(self.states)):
        self.states[i] += state[i]
    if len(self.next_states) == 0:
      for i in range(len(next_state)):
        self.next_states += [next_state[i]]
    else:
      for i in range(len(self.next_states)):
        self.next_states[i] += next_state[i]

    if is_end:
      if len(self.end_actions) > 2 * max_len:
        for i in range(len(self.end_states)):
          self.end_states[i] = self.end_states[i][max_len:]
        self.end_actions = self.end_actions[max_len:]
        self.end_rewards = self.end_rewards[man_len:]
        for i in range(len(self.end_next_states)):
          self.end_next_states[i] = self.end_next_states[i][max_len:]
        self.end_not_dones = self.end_not_dones[max_len:]
        self.end_steps = self.end_steps[max_len:]

      self.end_actions += [end_action]
      self.end_rewards += [end_reward]
      self.end_not_dones += [not end_done]
      self.end_steps += [end_step]
      if len(self.end_states) == 0:
        for i in range(len(end_state)):
          self.end_states += [end_state[i]]
      else:
        for i in range(len(self.end_states)):
          self.end_states[i] += end_state[i]
      if len(self.end_next_states) == 0:
        for i in range(len(end_next_state)):
          self.end_next_states += [end_next_state[i]]
      else:
        for i in range(len(self.end_next_states)):
          self.end_next_states[i] += end_next_state[i]

  def sample_end(self, n):
    assert len(self.end_actions) > 0 and n > 0
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
    return samp


  def sample_traj(self, n):
    assert len(self.actions) > 0 and n > 0
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
    return samp

  def sample(self, n, traj_end_ratio):
    assert traj_end_ratio < 1 and traj_end_ratio > 0, "traj_end_ratio must lie between 0 and 1"
    alpha = self.avg_traj_seg_len / (1/traj_end_ratio -1 + self.avg_traj_seg_len)
    end_states, end_actions, end_rewards, \
      end_next_states, end_not_dones, end_steps =  self.sample_end(min(int(alpha*n), 1))
    states, actions, rewards, \
      next_states, not_dones, steps = self.sample_traj(min(int((1-alpha)*n), 1))
    actions += end_actions
    rewards += end_rewards
    not_dones += end_not_dones
    steps += end_steps
    for i in len(states):
      states[i] += end_states[i]
      next_states[i] += end_next_states[i]
    return (states, actions, rewards, next_states, not_dones, steps)

  def __len__(self):
    l = len(self.actions)
    return l