
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np
from shooting_env import SimpleShootingEnv

def env_creator(env_config):
    return SimpleShootingEnv(discrete_action=True, num_angles=16, set_ManualMode=False)

register_env("SimpleShooting-v0", env_creator)

config = (
    PPOConfig()
    .environment(env="SimpleShooting-v0")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .training(train_batch_size=2000, model={"fcnet_hiddens": [128, 128]})
)

algo = config.build()

for i in range(10):  # Train for 10 iterations
    result = algo.train()
    print(f"Iteration {i}: reward={{result['episode_reward_mean']}}")

algo.save("ppo_simple_shooting")
