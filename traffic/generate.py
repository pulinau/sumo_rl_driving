#!python3
import subprocess
import random
import os

randomTrip = "~/project/sumo-bin/tools/randomTrips.py"
#mean = "4"
#" --binomial=" + mean + \

for i in range(3000, 4000):
    period = random.randrange(1, 4)/500
    temp0_file_name = "temp0.rou.xml"
    randomTripArg = " -n test.net.xml -r " + temp0_file_name + \
                    " -e 0.1" + \
                    " -p " + str(period) + \
                    ' --fringe-factor 10 --trip-attributes="departLane=\\"random\\" departSpeed=\\"random\\" departPos=\\"random_free\\""'   
    subprocess.run(randomTrip + randomTripArg, shell=True, check=True)
    
    os.remove("trips.trips.xml")

    period = random.randrange(1, 10)/5
    temp1_file_name = "temp1.rou.xml"
    randomTripArg = " -n test.net.xml -r " + temp1_file_name + \
                    ' --prefix 1_' + \
                    " -b 0.2 -e 640" + \
                    " -p " + str(period) + \
                    ' --fringe-factor 10 --trip-attributes="departLane=\\"random\\" departSpeed=\\"random\\" departPos=\\"base\\""'
    subprocess.run(randomTrip + randomTripArg, shell=True, check=True)
    
    with open(temp0_file_name, "r+") as fd_in0, open(temp1_file_name, "r+") as fd_in1, open("test"+str(i)+".rou.xml", "w+") as fd_out:
        contents0 = fd_in0.readlines()
        contents1 = fd_in1.readlines()

        contents = contents0[:-1] + contents1[33:]
        
        contents.insert(33, '    <vType id="EgoCar" color="0,1,0"/>\n')
        contents[34] = '    <vehicle id="ego" depart="0.00" departLane="random" departPos="random" departSpeed="random" type="EgoCar">\n'
        fd_out.writelines(contents)

