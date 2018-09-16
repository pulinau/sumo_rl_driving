import multiprocessing as mp

parent_conn_list, child_conn_list = zip(*[mp.Pipe() for _ in range(4)])

Simulation(conn, sumo_cfg)




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--play")
  args = parser.parse_args()

  env_list = [MultiObjSumoEnv(sumo_cfg) for i in range(6)]
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