from include import *
from sumo_gym import SumoCfg

# --------------------------
#          SUMO
# --------------------------
SUMO_TOOLS_DIR = "/home/ken/project/sumo-bin/tools"
SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo"
SUMO_CONFIG = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.sumocfg"
SUMO_TIME_STEP = 0.1

# "--net-file" and "route_file"
SUMO_CMD = [SUMO_BIN,
            "-c", "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.sumocfg",
            #"--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--collision.check-junctions", "true",
            "--step-length", str(SUMO_TIME_STEP)]
#            "--lanechange.duration", "2"]
NET_XML_FILE = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.net.xml"
ROU_XML_FILE_LIST = ["test0.rou.xml"]

EGO_VEH_ID = "ego"
MAX_VEH_ACCEL = 2.6
MAX_VEH_DECEL = 4.5
MAX_VEH_SPEED = 55.55

# --------------------------
#        observation
# --------------------------
NUM_LANE_CONSIDERED = 3 # number of lanes considered on each side of ego
NUM_VEH_CONSIDERED = 16
OBSERVATION_RADIUS = 600

# --------------------------
#         reward
# --------------------------
MAX_COMFORT_ACCEL_LEVEL = ActionAccel.MINACCEL
MAX_COMFORT_DECEL_LEVEL = ActionAccel.MINDECEL

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
               MAX_COMFORT_ACCEL_LEVEL,
               MAX_COMFORT_DECEL_LEVEL)