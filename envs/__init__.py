import gym
from envs.Humanoid import humanoid
gym.envs.register(
    id='Humanoid_Motion-v0',
    entry_point='envs:humanoid',
    max_episode_steps=100000,
)