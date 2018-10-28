#!python3
__author__ = "Changjian Li"

from include import *
from action import *
from observation import *
from sumo_gym import *
from dqn import *

import random
random.seed()
import multiprocessing as mp
import queue
from copy import deepcopy
import os

from sumo_cfgs import sumo_cfg

class returnX():
  def __init__(self, x):
    self.x = x

  def __call__(self, *args, **kwargs):
    return self.x
from collections import deque

class decreaseProb():
  def __init__(self, alpha, beta):
    self.alpha = alpha
    self.beta = beta

  def __call__(self, x):
    return 1 / (1 + np.exp(self.alpha * (x - beta)))

def run_env(sumo_cfg, dqn_cfg_list, obs_q_list, action_q_list, traj_q_list, play, max_ep, id):
  try:
    max_step = 3200
    env = MultiObjSumoEnv(sumo_cfg)

    violation_safety_hist = []
    violation_yield_hist = []
    violation_turn_hist = []

    for ep in range(max_ep):
      print("env id: {}".format(id), "episode: {}/{}".format(ep, max_ep))
      if play:
        init_step = 0
        model_index_list = [None] * len(dqn_cfg_list)
      else:
        init_step = random.randrange(60)
        model_index_list = [None] * len(dqn_cfg_list)
        for i in range(len(dqn_cfg_list)):
          if len(dqn_cfg_list[i].model_rst_prob_list) > 0:
            model_index_list[i] = random.randrange(len(dqn_cfg_list[i].model_rst_prob_list))
      obs_dict = env.reset(init_step)
      traj = []

      for step in range(max_step):
        violated_safety = False
        violated_yield = False
        violated_turn = False
        if step == 0:
          if play:
            env.agt_ctrl = True

        """
          else:
            if random.uniform(0, 1) < 0.1:
              env.agt_ctrl = False
        else:
          if not play:
            if random.uniform(0, 1) < 0.5:
              if env.agt_ctrl == False:
                env.agt_ctrl = True
              else:
                env.agt_ctrl = False
        """

        # select action
        if step == 0:
          if env.agt_ctrl == False:
            action = 0
            action_info = "sumo"
          else:
            action_set_list, sorted_idx_list = [], []

            for obs_q, model_index in zip(obs_q_list, model_index_list):
              obs_q.put((deepcopy(obs_dict), model_index))

            for action_q in action_q_list:
              while True:
                try:
                  (action_set, sorted_idx) = action_q.get(block=False)
                  break
                except queue.Empty:
                  continue
              action_set_list += [action_set]
              sorted_idx_list += [sorted_idx]

            is_explr_list = [False] * len(dqn_cfg_list)
            i = random.randrange(len(dqn_cfg_list))
            important = False
            if not play and random.random() < dqn_cfg_list[i].epsilon:
              is_explr_list[i] = True
              important = True

            action, action_info = select_action(dqn_cfg_list, is_explr_list, action_set_list, sorted_idx_list, 1)
        else:
          action = next_action
          action_info = next_action_info
          important = next_important

        if action == 7:
          action_full = 10
        elif action == 8:
          action_full = 17
        else:
          action_full = action
        next_obs_dict, (reward_list, done_list, violation_list), env_state, action_dict = \
          env.step({"lane_change": ActionLaneChange(action_full // len(ActionAccel)),
                   "accel_level": ActionAccel(action_full % len(ActionAccel))})
        action_full = action_dict["lane_change"].value * len(ActionAccel) + action_dict["accel_level"].value
        if action_full >= len(ActionAccel) and  action_full < 2 * len(ActionAccel):
          action = 7
        if action_full >= 2 * len(ActionAccel):
          action = 8

        violated_safety = violated_safety or violation_list[0]
        violated_yield = violated_yield or violation_list[1]
        violated_turn = violated_turn or violation_list[2]

        if True: # play:
          print("action: ", action)

        if env.agt_ctrl == False:
          action_info = "sumo"

        if step % 1 == 0 or action >= len(ActionAccel):

          # choose tentative actions for each objective
          for obs_q in obs_q_list:
            obs_q.put((deepcopy(next_obs_dict), None))

          action_set_list, sorted_idx_list = [], []
          tent_action_list = []
          tent_action_info_list = []

          for i, action_q in enumerate(action_q_list):
            while True:
              try:
                (action_set, sorted_idx) = action_q.get(block=False)
                break
              except queue.Empty:
                continue
            action_set_list += [action_set]
            sorted_idx_list += [sorted_idx]

            tent_action, tent_action_info = select_action(dqn_cfg_list[:i + 1], [False] * (i + 1), action_set_list,
                                                          sorted_idx_list, 1)
            tent_action_list += [tent_action]
            tent_action_info_list += [tent_action_info]

          # choose next action using model[model_index]
          action_set_list, sorted_idx_list = [], []

          for obs_q, model_index in zip(obs_q_list, model_index_list):
            obs_q.put((deepcopy(next_obs_dict), model_index))

          for action_q in action_q_list:
            while True:
              try:
                (action_set, sorted_idx) = action_q.get(block=False)
                break
              except queue.Empty:
                continue
            action_set_list += [action_set]
            sorted_idx_list += [sorted_idx]

          is_explr_list = [False] * len(dqn_cfg_list)
          next_important = False
          i = random.randrange(len(dqn_cfg_list))
          if not play and random.random() < dqn_cfg_list[i].epsilon:
            is_explr_list[i] = True
            next_important = True
          next_action, next_action_info = select_action(dqn_cfg_list, is_explr_list, action_set_list, sorted_idx_list,
                                                        1)

        if env_state != EnvState.DONE:
          traj.append((obs_dict, action, reward_list, next_obs_dict, tent_action_list, done_list, important))

        obs_dict = next_obs_dict

        if env_state == EnvState.DONE:
          prob = returnX(1)
          print("Sim ", id, " success, step: ", step)
          break
        if env_state != EnvState.NORMAL:
          prob = returnX(1)
          print("Sim ", id, " terminated, step: ", step, action_dict, action_info, reward_list, done_list, env_state,
                env.agt_ctrl)
          break
        if step == max_step - 1:
          prob = returnX(1)
          print("Sim ", id, " timeout, step: ", step)
          violated_yield = True
          break

      for i, traj_q in enumerate(traj_q_list):
        traj_q.put(
          ([deepcopy((obs_dict, action, reward_list[i], next_obs_dict, tent_action_list[i], done_list[i], important))
            for obs_dict, action, reward_list, next_obs_dict, tent_action_list, done_list, important in traj],
           prob))

      violation_safety_hist += [violated_safety]
      violation_yield_hist += [violated_yield]
      violation_turn_hist += [violated_turn]

  except:
    raise

  finally:
    f = open("result" + str(id), "a")
    f.writelines(["safety violation: " +  str(violation_safety_hist) + "\n"])
    f.writelines(["regulation violation (yield): " +  str(violation_yield_hist) + "\n"])
    f.writelines(["regulation violation (turn): " + str(violation_turn_hist) + "\n"])
    f.close()


def select_action(dqn_cfg_list, is_explr_list, action_set_list, sorted_idx_list, num_action, greedy=False):
  """
  Select an action based on the action choice of each objective.
  :param dqn_cfg_list:
  :param action_set_list: list of "good enough" actions of each objective
  :param explr_set_list: list of actions each objective want to explore
  :param sorted_idx_list: list of sorted actions based on (descending) desirability of each objective,
                          used in case there's no "good enough" action that satisfies all objectives
  :param num_action: the least num of action that's assumed to exist
  :return: action
  """
  valid = set(range(reduced_action_size))

  for action_set, is_explr, sorted_idx, dqn_cfg in zip(action_set_list, is_explr_list, sorted_idx_list, dqn_cfg_list):
    if is_explr:
      return (random.sample(valid, 1)[0], "explr: " + dqn_cfg.name)
    invalid = valid - action_set
    valid = valid & action_set

    if len(valid) < num_action:
      invalid = [(sorted_idx.index(x), x) for x in invalid]
      invalid = sorted(invalid)[:num_action - len(valid)]
      invalid = set([x[1] for x in invalid])
      invalid = [(x, "compromise: " + dqn_cfg.name) for x in invalid]
      break
    else:
      invalid = []

  if greedy:
    if len(valid) == 0:
      return invalid[0]
    else:
      valid = [(sorted_idx.index(x), x) for x in valid]
      valid = [(sorted(valid)[0][1], "greedy: " + dqn_cfg.name)]
      return valid[0]

  valid = [(x, "exploit: " + dqn_cfg.name) for x in valid]
  return random.sample(valid + invalid, 1)[0]

def run_QAgent(sumo_cfg, dqn_cfg, pretrain_traj_list, obs_q_list, action_q_list, traj_q_list, cuda_vis_devs):
  try:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_vis_devs
    agt = DQNAgent(sumo_cfg, dqn_cfg)

    ep = 0
    step = 0
    while True:
      for obs_q, action_q in zip(obs_q_list, action_q_list):
        try:
          obs_dict, model_index = obs_q.get(block=False)
          action_q.put(agt.select_actions(obs_dict, model_index=model_index))
          step += 1
        except queue.Empty:
          continue

      for traj_q in traj_q_list:
        try:
          traj, prob = traj_q.get(block=False)
          agt.remember(traj, prob)
        except queue.Empty:
          continue

      # if agt.name == 'regulation' or agt.name == 'safety':
      #  print("training ", agt.name, " episode: {}".format(ep))

      if random.random() < 1:
        agt.replay()

        if ep % 500 == 500 - 1:
          agt.update_target()
          agt.save_model()

        ep += 1

      if step % 40000 == 1:
        agt.save_model(suffix=str(step//40000))

  except:
    raise
