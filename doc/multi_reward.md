# multi_reward.py

`multi_reward.py` provides the multidimensional reward function for sumo environment. The reward consists of five dimensions:

* `r_safety`
* `r_regulation`
* `r_mission`
* `r_comfort`
* `r_curiosity`

with descending priority. The first four dimensions are hardcoded as part of the problem specification. The fifth dimention `r_curiosity` is a novelty based exploration for skill discovery.

## `r_safety`
Deal with hard safety constraints, which means keeping a safety distance at all time.
$|| (\mathbf{v}_e - \mathbf{v})t_{gap} + \mathbf{p}_e - \mathbf{p}||$

## `r_regulation`

## `r_mission`
help train the ego vehicle to learn the fastest way of getting to a destination

## `r_comfort`
Avoid jerk in speed. The reward is the negative of the speed variance in the last 10 steps

## `r_curiosity`
Novelty-based exploration. The reward is not hard-coded into the environment, rather, its generated dynamically depending on the ego vehicle's familiarity with the current state.
