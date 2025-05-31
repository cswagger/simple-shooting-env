import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from shooting_env import SimpleShootingEnv  # Make sure your env is named fixed_env.py or adjust this

# Create the environment with rgb_array mode for video recording
env = SimpleShootingEnv(discrete_action=True, num_angles=8, set_ManualMode=False)
env = RecordVideo(
    env,
    video_folder="videos",  # Make sure this folder exists or gets created
    episode_trigger=lambda episode_id: True,  # Record every episode
    name_prefix="shooting_test"
)

obs, _ = env.reset()
done = False
total_reward = 0

for _ in range(10000):  # 300 steps = 10 seconds at 30 fps
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        break

env.close()
print(f"Episode done. Total reward: {total_reward}")
