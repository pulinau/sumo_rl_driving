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

from sumo_cfgs import sumo_cfg
from dqn_cfgs import cfg_safety, cfg_regulation, cfg_comfort, cfg_speed

def run_env(sumo_cfg, dqn_cfg_list, end_q, obs_q_list, action_q_list, traj_q_list, play, max_ep, id):
  max_step = 2000
  env = MultiObjSumoEnv(sumo_cfg)

  for ep in range(max_ep):
    print("env id: {}".format(id), "episode: {}/{}".format(ep, max_ep))
    obs_dict = env.reset()
    traj = []

    for step in range(max_step):

      """
      if play:
        env.agt_ctrl = True
      elif step == 0:
        if random.uniform(0, 1) < 0.5:
          env.agt_ctrl = True
        else:
          env.agt_ctrl = False
      else:
        if random.uniform(0, 1) < 0.1:
          if env.agt_ctrl == True:
            env.agt_ctrl = False
          else:
            env.agt_ctrl = True
      """
      env.agt_ctrl = True

      # select action
      if env.agt_ctrl == False:
        action = 0
        action_info = "sumo"
      else:
        action_set_list, explr_set_list, sorted_idx_list = [], [], []

        for obs_q in obs_q_list:
          obs_q.put(obs_dict)

        for action_q in action_q_list:
          while action_q.empty():
            if not end_q.empty():
              return
          action_set, explr_set, sorted_idx = action_q.get()

          action_set_list += [action_set]
          explr_set_list += [explr_set]
          sorted_idx_list += [sorted_idx]

        action, action_info = select_action(dqn_cfg_list, action_set_list, explr_set_list)

      next_obs_dict, reward_list, env_state, action_dict = env.step(
        {"lane_change": ActionLaneChange(action // len(ActionAccel)), "accel_level": ActionAccel(action % len(ActionAccel))})
      action = action_dict["lane_change"].value * len(ActionAccel) + action_dict["accel_level"].value
      print(action, action_info)
      traj.append((obs_dict, action, reward_list, next_obs_dict, env_state != EnvState.NORMAL))

      obs_dict = next_obs_dict

      if env_state == EnvState.DONE:
        print("Sim ", id, " success, step: ", step)
        break
      if env_state != EnvState.NORMAL:
        print("Sim ", id, " terminated, step: ", step, action_dict, action_info, reward_list, env_state,
              env.agt_ctrl)
        break
      if step == max_step - 1:
        print("Sim ", id, " timeout, step: ", step)
        break

    for i, traj_q in enumerate(traj_q_list):
      traj_q.put([(obs_dict, action, reward_list[i], next_obs_dict, done)
                  for obs_dict, action, reward_list, next_obs_dict, done in traj])

  end_q.put(True)

def select_action(dqn_cfg_list, action_set_list, explr_set_list, sorted_idx_list):
  """
  Select an action based on the action choice of each objective. The underlying assumption is that there are at least 3
  actions that are satisfy all objectives relatively in a state.
  :param dqn_cfg_list:
  :param action_set_list: list of "good enough" actions of each objective
  :param explr_set_list: list of actions each objective want to explore
  :param sorted_idx_list: list of sorted actions based on desirability of each objective,
                          used in case there's no "good enough" action that satisfies all objectives
  :return: action
  """
  valid = action_set_list[0]
  invalid = explr_set_list[0]
  invalid_list = [(x, "explr: " + dqn_cfg_list[0].name) for x in invalid] # exploration actions with info

  for action_set, explr_set, sorted_idx, dqn_cfg in zip(action_set_list, explr_set_list, sorted_idx_list, dqn_cfg_list):
    if len(valid & action_set) < 3:
      new_invalid = [(sorted_idx.index(x), x) for x in valid - action_set]
      new_invalid = sorted(new_invalid)[:3 - len(valid & action_set)]
      new_invalid = [x[1] for x in new_invalid]
      invalid = invalid | set(new_invalid)
      invalid_list += [(x, "explr: " + dqn_cfg.name) for x in new_invalid]
      print("no available action for " + dqn_cfg.name)
      break
    new_invalid = explr_set & valid - invalid
    invalid = invalid | new_invalid
    invalid_list += [(x, "explr: " + dqn_cfg.name) for x in new_invalid]
    valid = valid & action_set

  valid = [(x, "exploit") for x in valid]
  return random.sample(valid + invalid_list, 1)[0]

def run_QAgent(sumo_cfg, dqn_cfg, pretrain_traj_list, end_q, obs_q_list, action_q_list, traj_q_list, max_ep):
  agt = DQNAgent(sumo_cfg, dqn_cfg)
  agt.pretrain(pretrain_traj_list, 1)

  for ep in range(max_ep):
    for obs_q, action_q in zip(obs_q_list, action_q_list):
      while obs_q.empty():
        if not end_q.empty():
          return
      obs_dict = obs_q.get()
      action_q.put(agt.select_actions(obs_dict))

    for traj_q in traj_q_list:
      try:
        agt.remember(traj_q.get(block=False))
      except queue.Empty:
        pass

    print("training ", agt.name, " episode: {}/{}".format(ep, max_ep))
    agt.replay()

    if ep % 100 == 100-1:
      agt.update_target()
      agt.save()

  end_q.put(True)