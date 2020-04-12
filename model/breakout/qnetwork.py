from tf_agents.networks.q_network import QNetwork
from tensorflow import keras
import tensorflow as tf
import numpy as np
from env import ENV

class DQN(ENV):
    def __init__(self):
        super().__init__()
        super()._setup()
        self.tf_env = super().wrap_env()
        self.preprocessing_layer = keras.layers.Lambda(
                                lambda obs: tf.cast(obs, np.float32) / 255.)
        self.conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
        self.fc_layer_params=[512]

    def model(self):
        q_net = QNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            preprocessing_layers=self.preprocessing_layer,
            conv_layer_params=self.conv_layer_params,
            fc_layer_params=self.fc_layer_params)
        return q_net

if __name__ == "__main__":
    dqn = DQN()
    dqn.model()