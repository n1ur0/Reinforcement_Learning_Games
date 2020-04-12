import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np
from . import data
class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2
