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
from workers import run_env, train

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--play")
  args = parser.parse_args()

  env = MultiObjSumoEnv(sumo_cfg)
  max_ep = 100
  sim_inst = 4
  if args.play:
    print("True")
    for dqn_cfg in [cfg_safety, cfg_regulation, cfg_comfort, cfg_speed]:
      dqn_cfg.play = True
    max_ep = 10
    sim_inst = 1

  dqn_cfg_list = [cfg_safety, cfg_regulation, cfg_comfort, cfg_speed]
  agt_list = [DQNAgent(sumo_cfg, dqn_cfg) for dqn_cfg in dqn_cfg_list]

  env_list = [mp.Process(target=run_env, args=(sumo_cfg, agt_list, args.play, max_ep, i, )) for i in range(sim_inst)]
  #train_list = [mp.Process(target=train, args=(agt, )) for i, agt in enumerate(agt_list)]
  [p.start() for p in env_list]
  #[p.start() for p in train_list]
  [p.join() for p in env_list]
  #[p.join() for p in train_list]