from include import *
from sumo_cfgs import *
from sumo_gym import *
from observation import *
from action import *

SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo-gui"

SUMO_CMD = [SUMO_BIN,
            #"-c", "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.sumocfg",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--collision.check-junctions", "true",
            "--xml-validation", "never",
            "--step-length", str(SUMO_TIME_STEP),
            "-n", NET_XML_FILE,
            "-r"]

sumo_cfg.SUMO_CMD = SUMO_CMD
env = MultiObjSumoEnv(sumo_cfg)
obs = env.reset(0)

env.agt_ctrl = False


for _ in range(600):
  obs, reward_list, env_state, action_dict = \
    env.step({"lane_change": ActionLaneChange.NOOP, "accel_level": ActionAccel.NOOP})
  if env_state != EnvState.NORMAL:
    env.reset(0)
  print(obs)