#!python3
__author__ = "Changjian Li"

import random
import time
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from copy import deepcopy

class ReplayMemory():
  """
  Replay memory for dynamic n-step look-ahead. Since the problem domain has very sparse reward,
  instead of using a fixed look-ahead time step, bootstrap is only done when the reward is actually received.
  Insignificant reward in between can be ignored by supplying a predicate, which is treated as 0 reward.
  """
  def __init__(self, max_len, name):
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
    self.next_actions =[]
    self.not_dones = []
    self.steps = []

    self.end_states = []
    self.end_actions = []
    self.end_rewards = []
    self.end_next_states = []
    self.end_next_actions = []
    self.end_not_dones = []
    self.end_steps = []

    self._size = 0

    self.lock.release()

    self.name = name
    self.avg_traj_seg_len = 0

  def add_traj(self, traj, end_pred, prob):
    """
    :param traj: list of state transitions
    :param end_pred: decide whether the reward is significant enough to be considered
    :return:
    """
    assert prob >= 0 and prob <= 1, "probability must be between 0 and 1"

    self.lock.acquire()

    traj_seg = []
    for i, (state, action, reward, next_state, next_action, done) in enumerate(traj[::-1]):
      if i == 0 or done or end_pred(reward):
        end_reward, end_state, end_next_action, end_done = deepcopy(reward), deepcopy(next_state), deepcopy(next_action), deepcopy(done)
        #if not done and i == 0:
        #  end_done = True
        if len(traj_seg) != 0:
          self.avg_traj_seg_len = (len(self.end_actions) * self.avg_traj_seg_len + len(traj_seg)) / \
                                  (len(self.end_actions) + 1)
          for j, x in enumerate(traj_seg):
            if random.uniform(0, 1) < prob:
              self._add(x, j == len(traj_seg)-1)
        traj_seg = []
        step = 0
      traj_seg.append(deepcopy((state, action, end_reward, end_state, end_next_action, end_done, step)))
      step += 1
    self.avg_traj_seg_len = (len(self.end_actions) * self.avg_traj_seg_len + len(traj_seg)) / \
                            (len(self.end_actions) + 1)
    for j, x in enumerate(traj_seg):
      if random.uniform(0, 1) < prob:
        self._add(x, j == len(traj_seg)-1)

    self.lock.release()

  def _add(self, tran, is_end):
    state, action, reward, next_state, next_action, done, step = deepcopy(tran)
    not_done = [[not y for y in x] for x in done]


    print("*************", self.name)
    print('actions: ', action)
    print('next_actions: ', next_action)
    print('rewards: ', reward)
    print('not_dones: ', not_done)
    print('steps: ', step)
    print('states: ', len(state[0]))
    print('next_states: ', len(next_state[0]))
    print("*************")

    try:
      assert len(self.actions) == len(self.next_actions) and \
             len(self.actions) == len(self.steps) and \
             (len(self.actions) == 0 or len(self.actions) == len(self.states[0])) and \
             (len(self.actions) == 0 or len(self.actions) == len(self.next_states[0])) and \
             (len(self.actions) == 0 or len(self.actions) == len(self.not_dones[0])) and \
             (len(self.actions) == 0 or len(self.actions) == len(self.rewards[0])), "must be of equal length"
    except:
      print('actions: ', self.actions)
      print('next_actions: ', self.next_actions)
      print('rewards: ', self.rewards)
      print('not_dones: ', self.not_dones)
      print('steps: ', self.steps)
      print('states: ', len(self.states[0]))
      print('next_states: ', len(self.next_states[0]))
      raise

    try:
      assert len(self.end_actions) == len(self.end_next_actions) and \
             len(self.end_actions) == len(self.end_steps) and \
             (len(self.end_actions) == 0 or len(self.end_actions) == len(self.end_states[0])) and \
             (len(self.end_actions) == 0 or len(self.end_actions) == len(self.end_next_states[0])) and \
             (len(self.end_actions) == 0 or len(self.end_actions) == len(self.end_not_dones[0])) and \
             (len(self.end_actions) == 0 or len(self.end_actions) == len(self.end_rewards[0])), "must be of equal length"
    except:
      print('end_actions: ', self.end_actions)
      print('end_next_actions: ', self.end_next_actions)
      print('end_rewards: ', self.end_rewards)
      print('end_not_dones: ', self.end_not_dones)
      print('end_steps: ', self.end_steps)
      print('end_states: ', len(self.end_states[0]))
      print('end_next_states: ', len(self.end_next_states[0]))
      raise

    if self._size > 2 * self.max_len + 2:
      for i in range(len(self.states)):
        self.states[i] = self.states[i][self.max_len:]
        self.next_states[i] = self.next_states[i][self.max_len:]
      for i in range(len(self.rewards)):
        self.rewards = self.rewards[i][self.max_len:]
      for i in range(len(self.not_dones)):
        self.not_dones = self.not_dones[i][self.max_len:]
      self.actions = self.actions[self.max_len:]
      self.next_actions = self.next_actions[self.max_len:]
      self.steps = self.steps[self.max_len:]
    self._size = len(self.actions)

    self.actions += [action]
    self.next_actions += [next_action]
    self.steps += [step]
    if self._size == 0:
      for i in range(len(state)):
        self.states += [state[i]]
        self.next_states += [next_state[i]]
      for i in range(len(reward)):
        self.rewards += [reward[i]]
      for i in range(len(not_done)):
        self.not_dones += [not_done[i]]
    else:
      for i in range(len(state)):
        self.states[i] += state[i]
        self.next_states[i] += next_state[i]
      for i in range(len(reward)):
        self.rewards[i] += reward[i]
      for i in range(len(not_done)):
        self.not_dones[i] += not_done[i]
    self._size += 1

    if is_end:
      # avoid using the same copy
      state, action, reward, next_state, next_action, done, step = deepcopy(tran)
      not_done = [[not y for y in x] for x in done]

      cap = int(self.max_len/(self.avg_traj_seg_len+1)) + 2
      if len(self.end_actions) > 2 * cap:
        for i in range(len(self.end_states)):
          self.end_states[i] = self.end_states[i][cap:]
          self.end_next_states[i] = self.end_next_states[i][cap:]
        for i in range(len(self.end_rewards)):
          self.end_rewards = self.end_rewards[i][cap:]
        for i in range(len(self.end_not_dones)):
          self.end_not_dones = self.end_not_dones[i][cap:]
        self.end_actions = self.end_actions[cap:]
        self.end_next_actions = self.end_next_actions[cap:]
        self.end_steps = self.end_steps[cap:]

      self.end_actions += [action]
      self.end_next_actions += [next_action]
      self.end_steps += [step]
      if len(self.end_states) == 0:
        for i in range(len(state)):
          self.end_states += [state[i]]
          self.end_next_states += [next_state[i]]
        for i in range(len(reward)):
          self.end_rewards += [reward[i]]
        for i in range(len(not_done)):
          self.end_not_dones += [not_done[i]]
      else:
        for i in range(len(state)):
          self.end_states[i] += state[i]
          self.end_next_states[i] += next_state[i]
        for i in range(len(reward)):
          self.end_rewards[i] += reward[i]
        for i in range(len(not_done)):
          self.end_not_dones[i] += not_done[i]

  def _sample_end(self, n):
    self.lock.acquire()
    
    assert n > 0, "sample size must be positive"
    assert len(self.end_actions) > 0, "replay memory empty"

    indices = random.sample(range(len(self.end_actions)), min(n, len(self.end_actions)))
    actions = [self.end_actions[i] for i in indices]
    next_actions = [self.end_next_actions[i] for i in indices]
    steps = [self.end_steps[i] for i in indices]
    rewards = [[x[i] for i in indices]
               for x in self.end_rewards]
    not_dones = [[x[i] for i in indices]
                 for x in self.end_not_dones]
    states = [[x[i] for i in indices]
              for x in self.end_states]
    next_states = [[x[i] for i in indices]
                   for x in self.end_next_states]
    samp = deepcopy((states, actions, rewards, next_states, next_actions, not_dones, steps))

    self.lock.release()
    return samp

  def _sample_traj(self, n):
    self.lock.acquire()

    assert n > 0, "sample size must be positive"
    assert self._size > 0, "replay memory empty"

    indices = random.sample(range(self._size), min(n, self._size))
    actions = [self.actions[i] for i in indices]
    next_actions = [self.next_actions[i] for i in indices]
    steps = [self.steps[i] for i in indices]
    rewards = [[x[i] for i in indices]
               for x in self.rewards]
    not_dones = [[x[i] for i in indices]
                 for x in self.not_dones]
    states = [[x[i] for i in indices]
              for x in self.states]
    next_states = [[x[i] for i in indices]
                   for x in self.next_states]
    samp = deepcopy((states, actions, rewards, next_states, next_actions, not_dones, steps))

    self.lock.release()
    return samp

  def sample(self, n, traj_end_ratio):
    assert traj_end_ratio < 1 and traj_end_ratio > 0, "traj_end_ratio must lie between 0 and 1"
    alpha = self.avg_traj_seg_len / (1/traj_end_ratio -1 + self.avg_traj_seg_len)
    assert alpha > 0 and alpha < 1, "alpha must be between 0 and 1"
    end_states, end_actions, end_rewards, \
      end_next_states, end_next_actions, end_not_dones, end_steps =  self._sample_end(max(int(alpha*n), 1))
    states, actions, rewards, \
      next_states, next_actions, not_dones, steps = self._sample_traj(max(int((1-alpha)*n), 1))
    actions += end_actions
    next_actions += end_next_actions
    steps += end_steps
    for i in range(len(states)):
      states[i] += end_states[i]
      next_states[i] += end_next_states[i]
    for i in range(len(rewards)):
      rewards[i] += end_rewards[i]
    for i in range(len(not_dones)):
      not_dones[i] += end_not_dones[i]
    return (states, actions, rewards, next_states, next_actions, not_dones, steps)

  def size(self):
    return self._size

class ReplayMemoryManager(BaseManager):
  pass
ReplayMemoryManager.register('ReplayMemory', ReplayMemory)