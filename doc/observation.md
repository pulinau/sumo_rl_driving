# observation.py
`observation.py` implements the observation space of SUMO OpenAI gym interface

## Observation Space
`observation_space` is a `spaces.Dict` object, whose key is mostly self explanatory. At each observation we consider a 
certain number (`NUM_LANE_CONSIDERED`) of vehicles around ego within the region of interest (ROI)

```
observation_space = spaces.Dict({"ego_speed": spaces.Box(0, env.MAX_VEH_SPEED, shape=(1,), dtype=np.float32),
             "ego_dist_to_end_of_lane": spaces.Box(0, env.OBSERVATION_RADIUS, shape=(1,), dtype=np.float32),
             "ego_in_intersection": spaces.Discrete(2),
             "ego_exists_left_lane": spaces.Discrete(2),
             "ego_exists_right_lane": spaces.Discrete(2),
             "ego_correct_lane_gap": spaces.Box(-env.NUM_LANE_CONSIDERED, env.NUM_LANE_CONSIDERED, shape=(1,), dtype=np.int16),
             "exists_vehicle": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "speed": spaces.Box(0, env.MAX_VEH_SPEED, (env.NUM_VEH_CONSIDERED,)),  # absolute speed
             "dist_to_end_of_lane": spaces.Box(0, env.OBSERVATION_RADIUS, (env.NUM_VEH_CONSIDERED,), dtype=np.float32),
             "in_intersection": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "relative_position": spaces.Box(-env.OBSERVATION_RADIUS, env.OBSERVATION_RADIUS, (env.NUM_VEH_CONSIDERED, 2), dtype=np.float32),
             "relative_heading": spaces.Box(-pi, pi, (env.NUM_VEH_CONSIDERED,), dtype=np.float32),
             "has_priority": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_peer": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_conflict": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_next": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_prev": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_left": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_right": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_ahead": spaces.MultiBinary(env.NUM_VEH_CONSIDERED),
             "veh_relation_behind": spaces.MultiBinary(env.NUM_VEH_CONSIDERED)
             })
```

`"veh relation"` describes the relation between the considered car and ego
| `"veh_relation"` name          | meaning                                                                 |
| ------------------------------ | ----------------------------------------------------------------------- |
| peer                           | shares the same next lane with ego                                      |
| conflict                       | approaching/in the same intersection and its route conflict with ego    |
| next                           | in the next lane of ego                                                 |
| prev                           | in the next lane of ego                                                 |
| left                           | to the left of ego                                                      |
| right                          | to the right of ego                                                     |
| ahead                          | in the same lane with ego and ahead                                     |
| behind                         | in the same lane with ego and behind                                    |

If we feed the observation space to a neural network, the weights connecting to the same key (of the observation space dictionary) 
should be identical. If a vehicle doesn't exist, all the relevant field is set to zero
