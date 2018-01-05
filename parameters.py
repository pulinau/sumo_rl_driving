SUMO_TOOLS_DIR = "/home/ken/project/sumo-bin/tools"
SUMO_BIN = "/home/ken/project/sumo-bin/bin/sumo-gui"
SUMO_CONFIG = "/home/ken/project/sumo-rl/test.sumocfg"
SUMO_TEME_STEP = 0.05
SUMO_CMD = [SUMO_BIN, "-c", SUMO_CONFIG, 
            "--time-to-teleport", "-1", 
            "--collision.action", "warn", 
            "--collision.check-junctions", "true",
            "--step-length", str(SUMO_TEME_STEP)]
#            "--lanechange.duration", "2"]
NET_XML_FILE = "/home/ken/project/sumo-rl/test0.net.xml"

EGO_VEH_ID = "ego"
MAX_VEH_ACCEL = 20
MAX_VEH_DECEL = 20
MAX_VEH_SPEED = 120

# --------------------------
#        observation
# --------------------------
NUM_LANE_CONSIDERED = 3 # number of lanes considered on each side of ego
NUM_VEHICLE_CONSIDERED = 16
OBSERVATION_RADIUS = 600
NUM_VEH_RELATION = len(["peer", 
                        "conflict", 
                        "conflict_intersection", 
                        "next", 
                        "left", 
                        "right", 
                        "ahead", 
                        "behind", 
                        "irrelevant"])
