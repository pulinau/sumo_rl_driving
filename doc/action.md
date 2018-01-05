# action.py
`action.py` implements the action space of SUMO OpenAI gym interface

## Action Space
`action_space` is a `spaces.Dict` object:

`"lane_change"` | `spaces.Discrete(3)`
--------------- | --------------------
noop            | 0
left            | 1
right           | 2

`"turn"` | `spaces.Discrete(3)`
-------- | --------------------
noop     | 0
left     | 1
right    | 2

`"accel_level"` | `spaces.Discrete(7)`
--------------- | --------------------
deccel          | 0-2
maintain        | 3
accel           | 4-6

Some combinations of actions are not always valid, namely:

* left/right turn with left/right lane change is invalid, because it's not clear what it means
* accel at maximum speed is treated as an noop, same for deccel at zero speed (no reverse support)

Invalid actions are treated as noop | maintain | noop
left/right turn is currently not implemented