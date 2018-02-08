#!python3
from action import *
from observation import *
from sumo_gym import *

# --------------------------
#          SUMO
# --------------------------
SUMO_TOOLS_DIR = "/home/ken/project/sumo-bin/tools"
SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo-gui"
SUMO_CONFIG = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.sumocfg"
SUMO_TIME_STEP = 0.1
SUMO_CMD = [SUMO_BIN, "-c", SUMO_CONFIG, 
            "--time-to-teleport", "-1", 
            "--collision.action", "warn", 
            "--collision.check-junctions", "true", 
            "--step-length", str(SUMO_TIME_STEP)]
#            "--lanechange.duration", "2"]
NET_XML_FILE = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.net.xml"

EGO_VEH_ID = "ego"
MAX_VEH_ACCEL = 20
MAX_VEH_DECEL = 20
MAX_VEH_SPEED = 120

# --------------------------
#        observation
# --------------------------
NUM_LANE_CONSIDERED = 3 # number of lanes considered on each side of ego
NUM_VEH_CONSIDERED = 16
OBSERVATION_RADIUS = 600

# --------------------------
#         reward
# --------------------------
MAX_COMFORT_ACCEL = 10
MAX_COMFORT_DECEL = 10

sumo_cfg = SumoCfg(
               # sumo
               SUMO_CMD, 
               SUMO_TIME_STEP, 
               NET_XML_FILE, 
               EGO_VEH_ID, 
               MAX_VEH_ACCEL, 
               MAX_VEH_DECEL, 
               MAX_VEH_SPEED, 
               # observation
               NUM_LANE_CONSIDERED, 
               NUM_VEH_CONSIDERED, 
               OBSERVATION_RADIUS, 
               # reward
               MAX_COMFORT_ACCEL, 
               MAX_COMFORT_DECEL)


env = MultiObjSumoEnv(sumo_cfg)
env.reset()

get_obs_dict(env)
obs_list, reward_list, done, info = env.step({"lane_change":ActionLaneChange.NOOP, "accel_level":ActionAccel.NOOP})
print(obs_list)

traci.close()
