from gymnasium.envs.registration import register

register(
    id="BR-v0",
    entry_point="gym_env_br.envs:BubbleRisingPositionEnv",
    kwargs={'parallel_envs': 0},
    max_episode_steps=500,  # ?
    reward_threshold=500.0,
)

