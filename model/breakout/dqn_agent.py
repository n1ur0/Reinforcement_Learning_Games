from tf_agents.agents.dqn.dqn_agent import DqnAgent
import tensorflow as tf
from tensorflow import keras
from qnetwork import DQN

class DQN_AGENT(DQN):
    def __init__(self):
        super().__init__()
        self.q_net = super().model()

    def train_model(self):
        train_step = tf.Variable(0)
        update_period = 4 # train the model every 4 steps
        optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
                                            epsilon=0.00001, centered=True)
        epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0, # initial ε
            decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
            end_learning_rate=0.01) # final ε
        agent = DqnAgent(self.tf_env.time_step_spec(),
                        self.tf_env.action_spec(),
                        q_network=self.q_net,
                        optimizer=optimizer,
                        target_update_period=2000, # <=> 32,000 ALE frames
                        td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                        gamma=0.99, # discount factor
                        train_step_counter=train_step,
                        epsilon_greedy=lambda: epsilon_fn(train_step))
        return agent

if __name__ == "__main__":
    dqn_agent = DQN_AGENT()
    agent = dqn_agent.train_model()
    agent.initialize()