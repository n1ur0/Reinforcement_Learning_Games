from tf_agents.environments import suite_gym

env = suite_gym.load("Breakout-v4")
print(env)