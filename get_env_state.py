 #!python3

SUMO_TOOLS_DIR = "/home/ken/project/sumo-0.30.0/tools"

try:
  sys.path.append(SUMO_TOOLS_DIR)
except ImportError:
  print("Please modify SUMO_TOOLS_DIR to the location of sumo tools")
import traci


