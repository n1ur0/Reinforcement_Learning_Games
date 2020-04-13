from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from env import ENV
from dqn_agent import DQN_AGENT
from replay import REPLAY
from logger import LOGS

class DRIVER(REPLAY, LOGS):

    def __init__(self):
        super().__init__()
        # replay_buffer = self._replay_buffer() 
        self.replay_buffer_observer = None
        LOGS().__init__()
        # self.training_metrics = LOGS()
        self.update_period = 4

    def driver(self):
        collect_driver = DynamicStepDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[self.replay_buffer_observer] + self.training_metrics,
            num_steps=self.update_period) # collect 4 steps for each training iterations
        return collect_driver
if __name__ == "__main__":
    driver = DRIVER()
    collect_driver = driver.driver()