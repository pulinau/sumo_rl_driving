#!python3
__author__ = "Changjian Li"

from include import *
from sumo_gym import SumoCfg

# --------------------------
#          SUMO
# --------------------------
SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo"
SUMO_TIME_STEP = 0.1

# "--net-file" and "route_file"
NET_XML_FILE = "/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test.net.xml"
ROU_XML_FILE_LIST = ["/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test" + str(i) + ".rou.xml" for i in range(4000)]
#ROU_XML_FILE_LIST = ["/home/ken/project/sumo-rl/sumo_openai_gym/traffic/test" + str(i) + ".rou.xml" for i in range(1)]
SUMO_CMD = [SUMO_BIN,
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--collision.mingap-factor", "0",
            "--collision.check-junctions", "true",
            "--xml-validation", "never",
            "--step-length", str(SUMO_TIME_STEP),
            "-n", NET_XML_FILE,
            #"--lanechange.duration", "2",
            "-r"]

EGO_VEH_ID = "ego"
MAX_VEH_ACCEL = 2.6
MAX_VEH_DECEL = 4.5
MAX_VEH_SPEED = 14

# --------------------------
#        observation
# --------------------------
NUM_LANE_CONSIDERED = 1 # number of lanes considered on each side of ego
NUM_VEH_CONSIDERED = 16
MAX_TTC_CONSIDERED = 60
OBSERVATION_RADIUS = 100

# --------------------------
#         reward
# --------------------------
MAX_COMFORT_ACCEL_LEVEL = ActionAccel.MINACCEL
MAX_COMFORT_DECEL_LEVEL = ActionAccel.MINDECEL

DEFAULT_COLOR = (255, 255, 0)
YIELD_COLOR = (255, 180, 0)

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