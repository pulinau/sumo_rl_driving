#!python3
__author__ = "Changjian Li"

from include import *
from sumo_cfgs import *
from utils import class_vars
import random
import tensorflow as tf
import numpy as np
from replay_mem import ReplayMemory, ReplayMemoryManager
from action import loosen_correct_actions
import time
import multiprocessing as mp
import queue
from collections import deque

class DQNCfg():
  def __init__(self, 
               name, 
               play,
               version,
               resume,
               state_size, 
               action_size,
               low_target,
               high_target,
               gamma,
               gamma_inc,
               gamma_max,
               epsilon,
               epsilon_dec,
               epsilon_min,
               threshold,
               memory_size,
               traj_end_pred,
               replay_batch_size,
               traj_end_ratio,
               _build_model,
               model_rst_prob_list,
               tf_cfg,
               reshape,
               _select_actions = None):
    self.name = name
    self.play = play # whether it's training or playing
    self.version = version
    self.resume = resume
    self.state_size = state_size
    self.action_size = action_size
    self.low_target = low_target
    self.high_target = high_target
    self.gamma = gamma
    self.gamma_inc = gamma_inc
    self.gamma_max = gamma_max
    self.epsilon = epsilon
    self.epsilon_dec = epsilon_dec
    self.epsilon_min = epsilon_min
    self.threshold = threshold
    self.memory_size = memory_size
    self.traj_end_pred = traj_end_pred
    self.replay_batch_size = replay_batch_size
    self.traj_end_ratio = traj_end_ratio # ratio states where bootstrap happens in the sample
    self._build_model = _build_model
    self.model_rst_prob_list = model_rst_prob_list
    self.tf_cfg = tf_cfg
    self.reshape = reshape
    self._select_actions = _select_actions

def feed_samp(replay_mem, samp_size, traj_end_ratio, samp_q):
  try:
    while True:
      if replay_mem.size() == 0:
        time.sleep(5)
        continue
      elif samp_q.qsize() < 80:
        samp_q.put(replay_mem.sample(samp_size, traj_end_ratio))
        # print("replay mem size: ", replay_mem.size())
  except:
    raise

class DQNAgent:
  def __init__(self, sumo_cfg, dqn_cfg):
    _attrs = class_vars(dqn_cfg)
    for _attr in _attrs:
      setattr(self, _attr, getattr(dqn_cfg, _attr))
    self.sumo_cfg = sumo_cfg

    if self._select_actions is not None:
      return

    assert self.memory_size >= self.replay_batch_size

    tf.keras.backend.set_session(tf.Session(config=self.tf_cfg))
    if self.play == True:
      self.epsilon = 0
    else:
      manager = ReplayMemoryManager()
      manager.start()
      self.memory = manager.ReplayMemory(self.memory_size, self.name)
      self.sample_q = mp.Queue(maxsize=100)
      self.feed_samp_p_list = [mp.Process(target=feed_samp,
                                          name='feed_samp ' + self.name,
                                          args=(self.memory,
                                                self.replay_batch_size,
                                                self.traj_end_ratio,
                                                self.sample_q))
                               for _ in range(1)]
      [p.start() for p in self.feed_samp_p_list]

      # half trained model to keep exploration steady
      self.model_list = [self._build_model() for _ in range(len(self.model_rst_prob_list))]
      self.target_model_list = [self._build_model() for _ in range(len(self.model_rst_prob_list))]
      self.loss_hist_list = [deque(maxlen=10) for _ in range(len(self.model_rst_prob_list))]

      self.loss_hist = deque(maxlen=10)

    if self.play == True:
      self.model = self._load_model(self.name + ".sav." + self.version)
    elif self.resume == True:
      self.model = self._load_model(self.name + ".sav." + self.version)
      self.target_model = self._load_model(self.name + ".sav." + self.version)
    else:
      self.model = self._build_model()
      self.target_model = self._build_model()

  def __del__(self):
    if self.play != True:
      [p.kill() for p in self.feed_samp_p_list]

  def remember(self, traj, prob):
    """remember experice with probability prob"""
    if self._select_actions is not None or self.play == True:
      return
    traj = [(self.reshape(obs_dict), action, reward, self.reshape(next_obs_dict), next_action, done, important)
            for obs_dict, action, reward, next_obs_dict, next_action, done, important in traj]
    self.memory.add_traj(traj, self.traj_end_pred, prob)

  def select_actions(self, obs_dict, model_index=None):
    """
    select actions based on Q value
    :param obs_dict:
    :return: (action_set, explore_set, sorted_idx) exploit action set, explore action set and a np.array of actions
              sorted according to descending desirability
    """
    if self._select_actions is not None:
      return self._select_actions(self.reshape(obs_dict))

    if model_index is None:
      model = self.model
    else:
      model = self.model_list[model_index]

    act_values = model.predict(self.reshape(obs_dict))[-1][0]
    sorted_idx = np.argsort(act_values)[::-1]

    if self.play == True:
      print(self.name, act_values)

    action_set = set(np.where(act_values >= np.max(act_values) + self.threshold)[0])

    return (action_set, list(sorted_idx))

  def replay(self):
    if self._select_actions is not None or self.play == True:
      return

    try:
      states, actions, rewards, next_states, next_actions, not_dones, steps = self.sample_q.get(block=False)
    except queue.Empty:
      # print("replay qsize: ", self.sample_q.qsize())
      # print(self.name, " empty")
      return

    for model_index, (model, target_model, loss_hist) in enumerate(zip(self.model_list + [self.model],
                                                                       self.target_model_list + [self.target_model],
                                                                       self.loss_hist_list + [self.loss_hist])):
      rewards = np.array(rewards)
      not_dones = np.array(not_dones)

      targets_f = model.predict_on_batch(states)
      m, n = len(targets_f) - 1, len(targets_f[0])

      actions = np.array(actions, dtype=np.int) * np.ones(shape=(m, n), dtype=np.int)
      next_actions = np.array(next_actions, dtype=np.int) * np.ones(shape=(m, n), dtype=np.int)
      steps = np.array(steps, dtype=np.int) * np.ones(shape=(m, n), dtype=np.int)

      next_q = target_model.predict_on_batch(next_states)[:-1]
      backup = np.array([[next_q[i][j][next_actions[i][j]] for j in range(n)] for i in range(m)])
      backup = (self.gamma ** (steps + 1)) * not_dones * backup

      # clamp targets larger than zero to zero
      backup[np.where(backup > 0)] = 0
      backup[np.where(backup < self.low_target)] = self.low_target

      targets = (self.gamma ** steps) * rewards + backup

      # clamp incorrect target to zero
      for i in range(m):
        for j in range(n):
          x = targets_f[i][j]
          x[np.where(x > 0)] = 0
          x[np.where(x < self.low_target)] = self.low_target
          x[actions[i][j]] = targets[i][j]

      # print(self.name, targets_f[-1][0])

      loss = model.train_on_batch(states, targets_f)
      if self.name == "safety":
        print("model ", model_index, " ", loss[0])

      loss_hist.append(loss[0])

      ep = 0
      factor = 1.2
      max_train_ep = loss[0]/max(np.median(loss_hist), 0.000000000001) - factor
      while loss[0] > factor * np.median(loss_hist) and ep < min(10 * max_train_ep, 10) and model_index == len(self.model_list):
        ep += 1
        targets_f = model.predict_on_batch(states)
        # clamp incorrect target to zero
        for i in range(m):
          for j in range(n):
            x = targets_f[i][j]
            x[np.where(x > 0)] = 0
            x[np.where(x < self.low_target)] = self.low_target
            x[actions[i][j]] = targets[i][j]
        loss = model.train_on_batch(states, targets_f)
        if self.name == "safety":
          print("model ", model_index, " ", self.name, " supplementary training:", np.median(loss_hist), loss[0])

    self.reset_models()
    if self.gamma < self.gamma_max:
      self.gamma += self.gamma_inc
    if self.epsilon > self.epsilon_min:
      self.epsilon -= self.epsilon_dec

  def update_target(self):
    if self._select_actions is not None or self.play == True:
      return
    for model, target_model in zip(self.model_list + [self.model],
                                   self.target_model_list + [self.target_model]):
      target_model.set_weights(model.get_weights())

  def reset_models(self):
    """
    reset exploration models
    :return: None
    """
    for i in range(len(self.model_list)):
      if random.random() < self.model_rst_prob_list[i]:
        self.model_list[i] = self._build_model()
        self.target_model_list[i] = self._build_model()

  def _load_model(self, filename):
    return tf.keras.models.load_model(filename, custom_objects={"tf": tf, "NUM_VEH_CONSIDERED": self.sumo_cfg.NUM_VEH_CONSIDERED})

  def save_model(self, suffix='current'):
    if self._select_actions is not None or self.play == True:
      return
    name = self.name + ".sav." + suffix
    self.model.save(name)
