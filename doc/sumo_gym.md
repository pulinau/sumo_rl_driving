# sumo_gym.py

**sumo_gym.py** implements an [OpenAI gym](https://github.com/openai/gym) interface of the [SUMO](http://sumo.dlr.de/wiki/Simulation_of_Urban_MObility_-_Wiki) traffic simulator for the behavioral planning of [autonomoose](http://www.autonomoose.net).

## Action Space
`action_space` is a `spaces.Dict` Object:
* `"lane_change"`: `spaces.Discrete(3)`
  * NOOP[0]
  * LEFT[1]
  * RIGHT[2]
* `"turn"`: `spaces.Discrete(3)`
  * NOOP[0]
  * LEFT[1]
  * RIGHT[2]
* `"speed_level"`: `spaces.Discrete(8)` 
  * BRAKEHARD[0]
  * BRAKE[1]
  * SPEED1-5[2-6]
  * MAXSPEED[7]

## Observation Space
`observation_space` is an occupancy grid of the environment centred at the ego vehicle.