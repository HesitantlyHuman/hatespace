from ray.tune import schedulers
from ray.tune import suggest
from ray.tune.progress_reporter import CLIReporter
from ray.tune.utils.util import date_str

import os

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from training import config, VAEBERT

from filelock import FileLock

epsilon = 1e-20
os.environ['TUNE_MAX_LEN_IDENTIFIER'] = '50'
    
def run_optuna_tune():
    config['dataset']['root_directory'] = os.getcwd()
    config['device'] = 'cuda:0'
    algorithm = HyperOptSearch(
    )
    algorithm = ConcurrencyLimiter(
        algorithm,
        max_concurrent = 6
    )
    scheduler = ASHAScheduler(
        max_t = config['training']['max_epochs'],
        grace_period = 3,
        reduction_factor = 2
    )
    reporter = CLIReporter(
        metric_columns = ['loss', 'precision', 'recall', 'accuracy', 'f1_score']
    )
    analysis = tune.run(
        VAEBERT,
        metric = 'loss',
        mode = 'min',
        resources_per_trial = {
            'cpu' : 4,
            'gpu' : 0
        },
        search_alg = algorithm,
        progress_reporter = reporter,
        scheduler = scheduler,
        num_samples = 100,
        config = config,
        local_dir = 'results',
        checkpoint_freq = 1,
        keep_checkpoints_num = 1,
        checkpoint_score_attr = 'accuracy'
    )

if __name__ == "__main__":
    run_optuna_tune()
