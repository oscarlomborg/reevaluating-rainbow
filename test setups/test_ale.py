import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

env = gym.make("ALE/DonkeyKong-v5", render_mode="human")

obs, info = env.reset()

print("Observation:", obs.shape)
print("Actions:", env.action_space)

for step in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()