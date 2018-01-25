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
$dist  = ||\mathbf{p}_e - \mathbf{p}||
if $dist > {safe\_dist}$, then there's no reward incurred. Otherwise, it grows quadratically with the decrease of distance
$r_{safety} = \alpha {(1-dist)}^2$
The hyperparameters are ${safe\_dist}$ and $\alpha$.

## `r_regulation`
Currently there are only a bunch of uncontrolled intersections in the road network, so only the following rules are included:

 * unless overtaking, stay in the correct lane 
 * ego car should yield if it's approaching the intersection from a low priority road

Which translates to the following concrete rules:
  
  * if the distance to intersection is less than 0.5 meters AND there's traffic on the relevant lane AND on low priority road,  negative reward is given for ego speed higher than 0
  * if the distance to intersection is less than 0.5 meters, negative reward for staying in the wrong lane (no lane change allowed within 0.5 meters from the intersection 
  * if there's no traffic ahead or beside in the correct lane, 

Some the regulations are difficult to encode in the form of traditional reward function, so we can consider making it rule based and output a series of acceptable actions, which is still compatible with the rest of the learning architecture

## `r_mission`
help train the ego vehicle to learn the fastest route to get to the destination (not included in the current implementation).

## `r_speed`
The vehicle should be moving as close to the maximum speed as possible if situation allows.

## `r_comfort`
Avoid frequent lane change and jerk in speed. The reward is the negative of the speed variance in the last 10 steps.

## `r_curiosity`
Novelty-based exploration. The reward is not hard-coded into the environment, rather, its generated dynamically depending on the ego vehicle's familiarity with the current implementation).
