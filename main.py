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
from workers import run_env, run_QAgent

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--play")
  args = parser.parse_args()

  env = MultiObjSumoEnv(sumo_cfg)
  max_ep = 50000
  sim_inst = 6
  dqn_cfg_list = [cfg_safety, cfg_regulation, cfg_comfort, cfg_speed]
  if args.play:
    print("True")
    for dqn_cfg in dqn_cfg_list:
      dqn_cfg.play = True
    max_ep = 10
    sim_inst = 1
  #"""
  with open("examples.npz", "rb") as file:
    npzfile = np.load(file)
    mem = npzfile[npzfile.files[0]]
    pretrain_traj_list = [[(obs_dict, action, None, None, True)] for traj in mem for obs_dict, action in traj]
    mem = None
    npzfile = None
  #"""
  #pretrain_traj_list = []

  obs_queues = [[mp.Queue() for j in range(sim_inst)] for i in range(len(dqn_cfg_list))]
  action_queues = [[mp.Queue() for j in range(sim_inst)] for i in range(len(dqn_cfg_list))]
  traj_queues = [[mp.Queue() for j in range(sim_inst)] for i in range(len(dqn_cfg_list))]
  end_q = mp.Queue() # if end_q is not empty, then all process must stop

  env_list = [mp.Process(target=run_env, args=(sumo_cfg,
                                               dqn_cfg_list,
                                               end_q,
                                               [obs_q[i] for obs_q in obs_queues],
                                               [action_q[i] for action_q in action_queues],
                                               [traj_q[i] for traj_q in traj_queues],
                                               args.play, max_ep, i,))
              for i in range(sim_inst)]


  agt_list = [mp.Process(target=run_QAgent, args=(sumo_cfg, dqn_cfg, pretrain_traj_list, end_q, obs_q_list, action_q_list, traj_q_list, 720*4*max_ep))
              for dqn_cfg, obs_q_list, action_q_list, traj_q_list in zip(dqn_cfg_list, obs_queues, action_queues, traj_queues)]

  [p.start() for p in env_list]
  [p.start() for p in agt_list]
  [p.join() for p in env_list]
  [p.join() for p in agt_list]
