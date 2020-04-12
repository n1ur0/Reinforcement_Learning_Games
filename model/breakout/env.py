# from tf_agents.environments import suite_gym
# from tf_agents.environments.wrappers import ActionRepeat
# from gym.wrappers import TimeLimit

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

from tf_agents.environments.tf_py_environment import TFPyEnvironment

# env = suite_gym.load("Breakout-v4")
# print(env.gym.get_action_meanings())


# repeating_env = ActionRepeat(env, times=4)

# limited_repeating_env = suite_gym.load(
#     "Breakout-v4",
#     gym_env_wrappers=[lambda env: TimeLimit(env, max_episode_steps=10000)],
#     env_wrappers=[lambda env: ActionRepeat(env, times=4)])


max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])


tf_env = TFPyEnvironment(env)

print(tf_env)