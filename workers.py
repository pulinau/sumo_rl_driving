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

def run_env(sumo_cfg, end_q, obs_q_list, action_q_list, traj_q_list, play, max_ep, id):
  max_step = 720
  env = MultiObjSumoEnv(sumo_cfg)

  for ep in range(max_ep):
    print("env id: {}".format(id), "episode: {}/{}".format(ep, max_ep))
    obs_dict = env.reset()
    traj = []

    for step in range(max_step):

      if play:
        env.agt_ctrl = True
      elif step == 0:
        if random.uniform(0, 1) < 0.5:
          env.agt_ctrl = True
        else:
          env.agt_ctrl = False
      else:
        if random.uniform(0, 1) < 0.03:
          if env.agt_ctrl == True:
            env.agt_ctrl = False
          else:
            env.agt_ctrl = True

      # select action
      for obs_q in obs_q_list:
        obs_q.put(obs_dict)

      action_set_list, explr_set_list = [], []
      for action_q in action_q_list:
        while action_q.empty():
          if not end_q.empty():
            return
        action_set, explr_set = action_q.get()

        action_set_list += [action_set]
        explr_set_list += [explr_set]

      action, action_info = select_action(action_set_list, explr_set_list)
      if env.agt_ctrl == False:
        action_info == "sumo"

      next_obs_dict, reward_list, env_state, action_dict = env.step(
        {"lane_change": ActionLaneChange(action // 7), "accel_level": ActionAccel(action % 7)})
      if env_state == EnvState.DONE:
        print("Ego ", id, " drove out of scene, step: ", step)
        break
      action = action_dict["lane_change"].value * 7 + action_dict["accel_level"].value

      traj.append((obs_dict, action, reward_list, next_obs_dict, env_state != EnvState.NORMAL))

      obs_dict = next_obs_dict

      if env_state != EnvState.NORMAL or step == max_step - 1:
        print("Simulation ", id, " terminated, step: ", step, action_dict, action_info, reward_list, env_state,
              env.agt_ctrl)
        break


    for i, traj_q in enumerate(traj_q_list):
      traj_q.put([(obs_dict, action, reward_list[i], next_obs_dict, done)
                  for obs_dict, action, reward_list, next_obs_dict, done in traj])

  end_q.put(True)

def select_action(action_set_list, explr_set_list):
  valid = set()
  invalid = [] # invalid stores the exploration actions

  for action_set, explr_set, name in zip(action_set_list, explr_set_list, ["safety", "regulation", "comfort", "speed"]):
    if len(valid) == 0:
      valid = valid | action_set
      invalid += [(x, "explr: " + name) for x in explr_set]
    else:
      invalid += [(x, "explr: " + name) for x in (explr_set & valid)]
    valid = valid & action_set
    if len(valid) == 0:
      #print("no available action for " + name)
      break

  valid = [(x, "valid") for x in valid]

  if len(valid) == 0 and len(invalid) == 0:
    return (random.sample(range(action_size), 1)[0], "exploration")
  else:
    return random.sample(valid + invalid, 1)[0]

def run_QAgent(sumo_cfg, dqn_cfg, end_q, obs_q_list, action_q_list, traj_q_list, max_ep):
  agt = DQNAgent(sumo_cfg, dqn_cfg)

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