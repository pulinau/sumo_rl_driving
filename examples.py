#!python3
__author__ = "Changjian Li"

from sumo_cfgs import sumo_cfg
from sumo_gym import *
import multiprocessing as mp
import random

def run_env(sumo_cfg, max_ep):
  id = random.randint(0, 65536)
  env = MultiObjSumoEnv(sumo_cfg)
  mem = []

  for ep in range(max_ep):
    print("env id: {}".format(id), "episode: {}/{}".format(ep, max_ep))
    obs_dict = env.reset()
    traj = []
    max_step = 2000

    for step in range(max_step):
      env.agt_ctrl = False

      action = 0
      next_obs_dict, _, env_state, action_dict = env.step(
        {"lane_change": ActionLaneChange(action // 7), "accel_level": ActionAccel(action % 7)})
      action = action_dict["lane_change"].value * 7 + action_dict["accel_level"].value
      print(action)

      traj.append((obs_dict, action))

      obs_dict = next_obs_dict

      if env_state == EnvState.DONE:
        print("Ego ", id, " drove out of scene, step: ", step)
        break
      if env_state != EnvState.NORMAL or step == max_step - 1:
        print("Ego ", id, " crashed, step: ", step)
        break

    mem.append(traj)

  env.close()
  return mem

if __name__ == "__main__":
  max_ep = 100
  num_sim = 10
  p= mp.Pool(num_sim)
  mem_list = p.starmap(run_env, [(sumo_cfg, max_ep)] * num_sim)
  mem = []
  for x in mem_list:
    mem += x

  with open("examples.npz", "wb+") as file:
    np.savez(file, mem)
    file.seek(0)