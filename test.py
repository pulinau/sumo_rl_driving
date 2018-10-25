from include import *
from sumo_cfgs import *
from sumo_gym import *
from observation import *
from action import *

SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo-gui"

NET_XML_FILE = "/home/ken/project/sumo-rl/sumo_openai_gym/roundabout/roundabout.net.xml"
ROU_XML_FILE_LIST = ["/home/ken/project/sumo-rl/sumo_openai_gym/roundabout/test" + str(i) + ".rou.xml" for i in range(1)]

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

sumo_cfg = SumoCfg(
               # sumo
               SUMO_CMD,
               SUMO_TIME_STEP,
               NET_XML_FILE,
               ROU_XML_FILE_LIST,
               EGO_VEH_ID,
               MAX_VEH_ACCEL,
               MAX_VEH_DECEL,
               MAX_VEH_SPEED,
               # observation
               NUM_LANE_CONSIDERED,
               NUM_VEH_CONSIDERED,
               MAX_TTC_CONSIDERED,
               OBSERVATION_RADIUS,
               # reward
               MAX_COMFORT_ACCEL_LEVEL,
               MAX_COMFORT_DECEL_LEVEL,
               DEFAULT_COLOR,
               YIELD_COLOR)

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