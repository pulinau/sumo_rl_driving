#!python3
__author__ = "Changjian Li"

from include import *
from sumo_cfgs import *
from utils import class_vars
import random
import tensorflow as tf
import numpy as np
from replay_mem import ReplayMemory
from action import loosen_correct_actions
import time
import multiprocessing as mp
import queue
import dill as pickle

class DQNCfg():
  def __init__(self, 
               name, 
               play, 
               state_size, 
               action_size,
               pretrain_low_target,
               pretrain_high_target,
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
               tf_cfg,
               reshape,
               _select_actions = None):
    self.name = name
    self.play = play # whether it's training or playing
    self.state_size = state_size
    self.action_size = action_size
    self.pretrain_low_target = pretrain_low_target
    self.pretrain_high_target = pretrain_high_target
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
    self.tf_cfg = tf_cfg
    self.reshape = reshape
    self._select_actions = _select_actions

def feed_samples(mem_q, sample_q, sample_size, traj_end_ratio):
  replay_mem = None
  while True:
    try:
      replay_mem = mem_q.get(block=False)
    except queue.Empty:
      if replay_mem is None:
        continue
      else:
        pass

    if len(replay_mem) == 0:
      print("replay_mem_len: ", len(replay_mem))
      time.sleep(0.1)
      continue
    elif sample_q.qsize() < 10:
      sample_q.put(replay_mem.sample(sample_size, traj_end_ratio))
      print("sample_queue size: ", sample_q.qsize())

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
      self.model = self._load_model(self.name + ".sav")
    else:
      self.memory = ReplayMemory(self.memory_size)
      self.mem_q, self.sample_q = mp.Queue(), mp.Queue()
      self.p = mp.Process(target=feed_samples, args=(self.mem_q, self.sample_q, self.replay_batch_size, self.traj_end_ratio))
      self.p.start()
      self.model = self._build_model()
      self.target_model = self._build_model()

  def remember(self, traj):
    if self._select_actions is not None or self.play == True:
      return
    traj = [(self.reshape(obs_dict), action, reward, self.reshape(next_obs_dict), done)
            for obs_dict, action, reward, next_obs_dict, done in traj]
    self.memory.add_traj(traj, self.traj_end_pred)

  def select_actions(self, obs_dict):
    """
    select actions based on Q value
    :param obs_dict:
    :return: (action_set, explore_set, sorted_idx) exploit action set, explore action set and a np.array of actions
              sorted according to descending desirability
    """
    if self._select_actions is not None:
      return self._select_actions(self.reshape(obs_dict))

    act_values = self.model.predict(self.reshape(obs_dict))[0]

    sorted_idx = np.argsort(act_values)[::-1]

    action_set = set(np.where(act_values > self.threshold)[0]) | set(sorted_idx[:6])
    explore_set = set([action for action in range(self.action_size) if np.random.rand() <= self.epsilon])
    explore_set = explore_set - action_set

    return (action_set, explore_set, list(sorted_idx))

  def pretrain(self, traj_list, ep):
    if self._select_actions is not None or self.play == True or len(traj_list) == 0:
      return

    try:
      self._load_model("pretrain_" + self.name + ".sav")
    except:
      self.pretrain_mem = ReplayMemory(end_pred=True)
      for traj in traj_list:
        traj = [(self.reshape(obs_dict), action, None, None, done)
                for obs_dict, action, reward, next_obs_dict, done in traj]
        self.pretrain_mem.add_traj(traj)

      states = [s[0][0] for s in self.pretrain_mem.traj_mem]
      actions = [s[0][1] for s in self.pretrain_mem.traj_mem]

      state_idx, correct_actions = loosen_correct_actions(actions)

      temp = []
      for i in range(len(states[0])):
        arr = [x[i][0] for x in states]
        temp += [arr]
      states = temp

      targets_f = self.pretrain_low_target * np.ones((len(states[0]), self.action_size))
      targets_f[state_idx, correct_actions] = self.pretrain_high_target

      self.model.fit(states, targets_f, epochs = ep)
      self.pretrain_mem = None
      self.save_model(name="pretrain_" + self.name + ".sav")

  def replay(self):
    if self._select_actions is not None or self.play == True or len(self.memory) == 0:
      return

    try:
      states, actions, rewards, next_states, not_dones, steps = self.sample_q.get(block=False)
    except queue.Empty:
      print("empty")
      return

    backup = (self.gamma**(steps+1)) * np.array(not_dones) * np.amax(self.target_model.predict_on_batch(next_states), axis = 1)

    # clamp targets larger than zero to zero
    backup[np.where(backup > 0)] = 0

    targets = (self.gamma**steps)*rewards + backup
    targets_f = self.target_model.predict_on_batch(states)
    targets_f[np.arange(targets_f.shape[0]), actions] = targets

    print(self.name, targets_f[0])
    #print(self.name , " training starts", time.time(), flush = True)

    if random.uniform(0, 1) < 0.01:
      id = random.randint(0, 65536)
      print(self.name, " id:", id, targets_f[0])
      for i in range(2):
        print(self.name, " id:", id, " ep:", i, "training loss: ", self.model.train_on_batch(states, targets_f))

    #print(self.name, " training ends", time.time(), flush = True)

    if self.gamma < self.gamma_max:
      self.gamma += self.gamma_inc
    if self.epsilon > self.epsilon_min:
      self.epsilon -= self.epsilon_dec

  def send_memory(self):
    if self._select_actions is not None or self.play == True:
      return
    self.mem_q.put(self.memory)

  def update_target(self):
    if self._select_actions is not None or self.play == True:
      return
    self.target_model.set_weights(self.model.get_weights())

  def _load_model(self, filename):
    return tf.keras.models.load_model(filename, custom_objects={"tf": tf, "NUM_VEH_CONSIDERED": self.sumo_cfg.NUM_VEH_CONSIDERED})

  def save_model(self, name=None):
    if self._select_actions is not None or self.play == True:
      return
    if name is None:
      name = self.name + ".sav"
    self.model.save(name)
