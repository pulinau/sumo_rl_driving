# Multi-Objective Q Learning

This documentation describes Multi-Objective Q learning algorithm and its implementation for self-driving cars simulated in SUMO

## Algorithm

## Implementation
[sumo_gym.py](sumo_gym.md) provides the standard [OpenAI gym](https://github.com/openai/gym) interface for RL training. However, in Multi-Reward Q learning, we need a list of rewards of different priorities, each for the training of one Q learning agent. Then we adopt an action selection strategy to enforce priority and to make explorations.

#### `class SumoEnvMultiR(SumoEnv)`
Child class of `class SumoEnv(gym.Env)`. Its `step` method overrides `SumoEnv.step` providing the same functionality, but returns a list of rewards and the corresponding list of observations as opposed to a single reward and observation.

#### `config.py`
