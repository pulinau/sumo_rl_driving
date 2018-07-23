#!python3
__author__ = "Changjian Li"

from include import *
from action import *
from observation import *
from sumo_gym import *
from dqn import *

import random
import multiprocessing as mp
import queue
from copy import deepcopy

from sumo_cfgs import sumo_cfg

def run_env(sumo_cfg, dqn_cfg_list, end_q, obs_q_list, action_q_list, traj_q_list, play, max_ep, id):
  max_step = 2000
  env = MultiObjSumoEnv(sumo_cfg)

  for ep in range(max_ep):
    print("env id: {}".format(id), "episode: {}/{}".format(ep, max_ep))
    obs_dict = env.reset()
    traj = []

    for step in range(max_step):

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
      if env.agt_ctrl == False:
        action = 0
        action_info = "sumo"
      else:
        action_set_list, explr_set_list, sorted_idx_list = [], [], []

        for obs_q in obs_q_list:
          obs_q.put((deepcopy(obs_dict), None))

        for action_q in action_q_list:
          while action_q.empty():
            if not end_q.empty():
              return
          action_set, explr_set, sorted_idx = action_q.get()

          action_set_list += [action_set]
          explr_set_list += [explr_set]
          sorted_idx_list += [sorted_idx]

        action, action_info = select_action(dqn_cfg_list, action_set_list, explr_set_list, sorted_idx_list, 3)

      next_obs_dict, reward_list, env_state, action_dict = env.step(
        {"lane_change": ActionLaneChange(action // len(ActionAccel)), "accel_level": ActionAccel(action % len(ActionAccel))})
      action = action_dict["lane_change"].value * len(ActionAccel) + action_dict["accel_level"].value
      #print(action, action_info)
      # choose tentative next action
      for obs_q in obs_q_list:
        obs_q.put((deepcopy(next_obs_dict), 0))

      action_set_list, explr_set_list, sorted_idx_list = [], [], []
      # next action list is only the tentative actions for training purpose, not the one actually taken
      next_action_list = []

      for i, action_q in enumerate(action_q_list):
        while action_q.empty():
          if not end_q.empty():
            return
        action_set, explr_set, sorted_idx = action_q.get()

        action_set_list += [action_set]
        explr_set_list += [explr_set]
        sorted_idx_list += [sorted_idx]

        tent_action, tent_action_info = select_action(dqn_cfg_list[:i + 1], action_set_list, explr_set_list, sorted_idx_list, 1, greedy=True)
        next_action_list += [tent_action]

      if env_state != EnvState.DONE:
        traj.append((obs_dict, action, reward_list, next_obs_dict, next_action_list, env_state != EnvState.NORMAL))

      obs_dict = next_obs_dict

      if env_state == EnvState.DONE:
        prob = 1
        print("Sim ", id, " success, step: ", step)
        break
      if env_state != EnvState.NORMAL:
        prob = 1
        print("Sim ", id, " terminated, step: ", step, action_dict, action_info, reward_list, env_state,
              env.agt_ctrl)
        break
      if step == max_step - 1:
        prob = 1
        print("Sim ", id, " timeout, step: ", step)
        break

    for i, traj_q in enumerate(traj_q_list):
      traj_q.put(([deepcopy((obs_dict, action, reward_list[i], next_obs_dict, next_action_list[i], done))
                   for obs_dict, action, reward_list, next_obs_dict, next_action_list, done in traj],
                  prob))

  end_q.put(True)

def select_action(dqn_cfg_list, action_set_list, explr_set_list, sorted_idx_list, num_action, greedy=False):
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
  valid = action_set_list[0]

  for action_set, explr_set, sorted_idx, dqn_cfg in zip(action_set_list, explr_set_list, sorted_idx_list, dqn_cfg_list):
    if len(explr_set) != 0:
      return (random.sample(explr_set, 1)[0], "explr: " + dqn_cfg.name)
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

  valid = [(x, "exploit") for x in valid]
  return random.sample(valid + invalid, 1)[0]

def run_QAgent(sumo_cfg, dqn_cfg, pretrain_traj_list, end_q, obs_q_list, action_q_list, traj_q_list):
  agt = DQNAgent(sumo_cfg, dqn_cfg)
  agt.pretrain(pretrain_traj_list, 1)

  ep = 0
  while True:
    for obs_q, action_q in zip(obs_q_list, action_q_list):
      try:
        obs_dict, epsilon = obs_q.get(block=False)
        action_q.put(agt.select_actions(obs_dict, epsilon))
      except queue.Empty:
        if not end_q.empty():
          return
        else:
          continue

    for traj_q in traj_q_list:
      try:
        traj, prob = traj_q.get(block=False)
        agt.remember(traj, prob)
      except queue.Empty:
        continue

    #if agt.name == 'regulation' or agt.name == 'safety':
    #  print("training ", agt.name, " episode: {}".format(ep))
    agt.replay()

    if ep % 1000 == 1000-1:
      agt.update_target()
      agt.save_model()

    ep += 1