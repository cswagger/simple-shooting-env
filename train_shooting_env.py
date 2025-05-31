import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from shooting_env import SimpleShootingEnv
from tqdm import tqdm

# Register custom environment
def env_creator(env_config):
    return SimpleShootingEnv(discrete_action=True, num_angles=16, set_ManualMode=False)

register_env("SimpleShooting-v0", env_creator)

# Configure PPO
config = (
    PPOConfig()
    .environment(env="SimpleShooting-v0")
    .framework("torch")
    .env_runners(
        num_env_runners=1,
        rollout_fragment_length=200
    )
    .rl_module(model_config={"fcnet_hiddens": [128, 128]})
    .training(train_batch_size=20000)
    .resources(num_gpus=1)
)

# Build the algorithm
algo = config.build()

# Training loop
for i in tqdm(range(5000)):
    result = algo.train()

# Save model
algo.save("/root/simple-shooting-env/ppo_simple_shooting")
