#!python3
__author__ = "Changjian Li"

import argparse

from include import *
from action import *
from observation import *
from sumo_gym import *
from dqn import *

import random
import multiprocessing as mp

from sumo_cfgs import sumo_cfg
from dqn_cfgs import cfg_safety, cfg_regulation, cfg_comfort, cfg_speed



def select_action(action_set_list, explr_set_list, dqn_cfg_list):
  valid = set()
  invalid = [] # invalid stores the exploration actions

  for action_set, explr_set, name in zip(action_set_list, explr_set_list, [dqn_cfg.name for dqn_cfg in dqn_cfg_list]):
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

def Qlearning(conn, sumo_cfg, dqn_cfg):
  agt = DQNAgent(sumo_cfg, dqn_cfg)

  while conn.recv() == True:
    obs_dict = conn.recv()
    state = agt.reshape(obs_dict)

    action_set, explr_set = agt.select_actions(state)
    conn.send((action_set, explr_set))

    next_obs_dict, reward, env_state, action_dict = conn.recv()
    action = action_dict["lane_change"].value * 7 + action_dict["accel_level"].value
    next_state = agt.reshape(next_obs_dict)
    agt.remember(state, action, reward, next_state, env_state != EnvState.NORMAL)

    agt.replay()

    if conn.recv() == True:
      agt.update_target()
      agt.save()

  conn.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--play")
  args = parser.parse_args()

  env = MultiObjSumoEnv(sumo_cfg)
  EPISODES = 60000
  if args.play:
    print("True")
    for dqn_cfg in [cfg_safety, cfg_regulation, cfg_comfort, cfg_speed]:
      dqn_cfg.play = True
    EPISODES = 10

  dqn_cfg_list = [cfg_safety, cfg_regulation, cfg_comfort, cfg_speed]
  parent_conn_list, child_conn_list = zip(*[mp.Pipe() for _ in range(4)])

  p_list = [mp.Process(target=Qlearning, args=(conn, sumo_cfg, dqn_cfg)) for conn, dqn_cfg in
            zip(child_conn_list, dqn_cfg_list)]
  [p.start() for p in p_list]

  for e in range(EPISODES):
    print("episode: {}/{}".format(e, EPISODES))
    obs_dict = env.reset()

    for step in range(6400):
      # env.agt_ctrl = False

      if args.play:
        env.agt_ctrl = True
      elif step == 0:
        if random.uniform(0, 1) < 0.5:
          env.agt_ctrl = True
        else:
          env.agt_ctrl = False
      else:
        if random.uniform(0, 1) < 0.01:
          if env.agt_ctrl == True:
            env.agt_ctrl = False
          else:
            env.agt_ctrl = True


      [conn.send(True) for conn in parent_conn_list]

      # send obs_dict
      [conn.send(obs_dict) for conn in parent_conn_list]

      import time
      print("entering: ", time.time())
      # select action
      action_set_list, explr_set_list = zip(*[conn.recv() for conn in parent_conn_list])
      action, action_info = select_action(action_set_list, explr_set_list, dqn_cfg_list)
      if env.agt_ctrl == False:
        action_info == "sumo"
      print("exiting: ", time.time())

      # act
      next_obs_dict, reward_list, env_state, action_dict = env.step(
        {"lane_change": ActionLaneChange(action // 7), "accel_level": ActionAccel(action % 7)})
      if env_state == EnvState.DONE:
        print("Ego successfully drived out of scene, step: ", step)

      [conn.send((next_obs_dict, reward, env_state, action_dict)) for conn, reward in
       zip(parent_conn_list, reward_list)]

      # save model
      if step % 100 == 1:
        [conn.send(True) for conn in parent_conn_list]
      else:
        [conn.send(False) for conn in parent_conn_list]

      obs_dict = next_obs_dict

      if env_state != EnvState.NORMAL or step == 6400 - 1:
        print("Simulation Terminated, step: ", step, action_dict, action_info, reward_list, env_state, env.agt_ctrl)
        break

  [conn.send(False) for conn in parent_conn_list]
  [p.join() for p in p_list]