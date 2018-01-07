# observation.py
`observation.py` implements the observation space of SUMO OpenAI gym interface

## Observation Space
`observation_space` is a `spaces.Dict` object, whose key is mostly self explanatory. At each observation we consider a 
certain number (`NUM_LANE_CONSIDERED`) of vehicles around ego within the region of interest (ROI)

```
spaces.Dict({"ego_speed": spaces.Box(0, MAX_VEH_SPEED, shape=(1,)),
             "ego_dist_to_end_of_lane": spaces.Box(0, float("inf"), shape=(1,)),
             "ego_in_intersection": spaces.Discrete(2),
             "ego_exists_left_lane": spaces.Discrete(2),
             "ego_exists_right_lane": spaces.Discrete(2),
             "ego_correct_lane_gap": spaces.Box(float("-inf"), float("inf"), shape=(1,)),
             "exists_vehicle": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "speed": spaces.Box(0, MAX_VEH_SPEED, (NUM_VEHICLE_CONSIDERED,)),  # absolute speed
             "dist_to_end_of_lane": spaces.Box(0, float("inf"), (NUM_VEHICLE_CONSIDERED,)),
             "in_intersection": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "relative_position": spaces.Box(-OBSERVATION_RADIUS, OBSERVATION_RADIUS, (NUM_VEHICLE_CONSIDERED, 2)),
             "relative_heading": spaces.Box(-pi, pi, (NUM_VEHICLE_CONSIDERED,)),
             "has_priority": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "veh_relation": spaces.MultiBinary(NUM_VEH_RELATION * NUM_VEHICLE_CONSIDERED)
             })
```

`"veh relation"` describes the relation between the considered car and ego
| `"veh_relation"` name          | index  | meaning                                                                 |
| ------------------------------ | ------ | ----------------------------------------------------------------------- |
| peer                           | 0      | shares the same next lane with ego                                      |
| conflict                       | 1      | approaching/in the same intersection and its route conflict with ego    |
| next                           | 3      | in the next lane of ego                                                 |
| left                           | 4      | to the left of ego                                                      |
| right                          | 5      | to the right of ego                                                     |
| ahead                          | 6      | in the same lane with ego and ahead                                     |
| behind                         | 7      | in the same lane with ego and behind                                    |
| irrelevant                     | 8      | none of the above, e.g. opposite lane/in other intersection/no conflict |

A vehicle can have several of the above relations with ego at the same time.

If we feed the observation space to a neural network, the weights connecting to the same key (of the observation space dictionary) 
should be identical. If a vehicle doesn't exist, all the relevant field is set to zero
