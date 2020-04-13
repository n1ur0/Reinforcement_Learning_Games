from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging

class LOGS():
    def __init__(self):
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

    def show_logs(self):
        logging.getLogger().setLevel(logging.INFO)
        log_metrics(self.train_metrics)
