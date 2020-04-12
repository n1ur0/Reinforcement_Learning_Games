
from dqn_agent import DQN_AGENT
from tf_agents.replay_buffers import tf_uniform_replay_buffer

class REPLAY(DQN_AGENT):
    def __init__(self):
        super().__init__()
        self.agent = super().train_model()
    def _replay_buffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec, # specification of data
            batch_size=self.tf_env.batch_size, # number of tragetories that will be added
            max_length=1000000)
        return replay_buffer

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

if __name__ == "__main__":
    buffer = REPLAY()
    replay_buffer = buffer._replay_buffer()
    replay_buffer_observer = replay_buffer.add_batch