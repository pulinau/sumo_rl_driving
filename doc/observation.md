# observation.py
`observation.py` implements the observation space of SUMO OpenAI gym interface

## Observation Space
`observation_space` is a `spaces.Dict` object, whose key is mostly self explanatory. At each observation we consider a 
certain number (`NUM_LANE_CONSIDERED`) of vehicles around ego within the region of interest (ROI)

```
spaces.Dict({"ego_speed": spaces.Box(0, MAX_VEHICLE_SPEED),
             "ego_dist_to_end_of_lane": spaces.Box(0, float("inf")),
             "ego_in_intersection": spaces.Discrete(2),
             "ego_exists_left_lane": spaces.Discrete(2),
             "ego_exists_right_lane": spaces.Discrete(2),
             "ego_correct_lane": spaces.Discrete(2 * NUM_LANE_CONSIDERED + 1),
             "exists_vehicle": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "speed": spaces.Box(0, MAX_VEHICLE_SPEED, (NUM_VEHICLE_CONSIDERED,)),  #absolute speed
             "relative_position": spaces.Box(float("-inf"), float("inf"), (NUM_VEHICLE_CONSIDERED, 2)),
             "relative_heading": spaces.Box(-pi, pi, (NUM_VEHICLE_CONSIDERED,)),
             "has_priority": spaces.MultiBinary(NUM_VEHICLE_CONSIDERED),
             "dist_to_end_of_lane": spaces.Box(0, float("inf"), (NUM_VEHICLE_CONSIDERED,)),
             "veh_relation": spaces.MultiBinary(NUM_VEH_RELATION * NUM_VEHICLE_CONSIDERED)
             })
```

`"veh relation"` describes the relation between the considered car and ego
| `"veh_relation"` name          | index  | meaning                                                                 |
| ------------------------------ | ------ | ----------------------------------------------------------------------- |
| peer                           | 0      | shares the same next lane with ego                                      |
| conflict                       | 1      | approaching the same intersection and its route conflict with ego       |
| conflict_intersection          | 2      | same as above except that the vehicle is already in intersection        |
| next                           | 3      | in next lane of ego                                                     |
| prev                           | 4      | in previous lane of ego                                                 |
| left                           | 5      | to the left of ego                                                      |
| right                          | 6      | to the right of ego                                                     |
| ahead                          | 7      | on the same lane with ego and ahead                                     |
| behind                         | 8      | on the same lane with ego and behind                                    |
| irrelevant                     | 9      | none of the above, e.g. opposite lane/in other intersection/no conflict |

A vehicle can have several of the above relations with ego at the same time.

If we feed the observation space to a neural network, the weights connecting to the same key (of the observation space dictionary) 
should be identical. If a vehicle doesn't exist, all the relevant field is set to zero