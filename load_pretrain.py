#!python3
__author__ = "Changjian Li"

import numpy as np
from replay_mem import ReplayMemory
from dqn import *
from dqn_cfgs import *
from sumo_cfgs import sumo_cfg

def load_pretrain():
  agt = DQNAgent(sumo_cfg, cfg_safety)

  with open("examples.npz", "rb") as file:
    npzfile = np.load(file)
    mem = npzfile[npzfile.files[0]]
    npzfile = None

  for traj in mem:
    for obs_dict, action, next_obs_dict in traj:
      agt.remember([(obs_dict, action, 0, next_obs_dict, True)])

  print(agt.memory.traj_mem)

load_pretrain()