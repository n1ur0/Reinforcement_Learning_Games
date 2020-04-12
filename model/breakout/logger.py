from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

logging.get_logger().set_level(logging.INFO)
log_metrics(train_metrics)