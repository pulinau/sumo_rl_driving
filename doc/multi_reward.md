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
$m  = ||\mathbf{p}_e - \mathbf{p}|| - \frac{(\mathbf{v}_e - \mathbf{v})(\mathbf{p}_e - \mathbf{p})t_{gap}}{||\mathbf{p}_e - \mathbf{p}||}$
if $m > 0$, then there's no reward incurred. Otherwise, it grows exponentially to negative infinity
$r_{safety} = e^{\alpha  m}$
The hyperparameters are $t_{gap}$ and $\alpha$

## `r_regulation`

## `r_mission`
help train the ego vehicle to learn the fastest way of getting to a destination

## `r_comfort`
Avoid jerk in speed. The reward is the negative of the speed variance in the last 10 steps

## `r_curiosity`
Novelty-based exploration. The reward is not hard-coded into the environment, rather, its generated dynamically depending on the ego vehicle's familiarity with the current state.
