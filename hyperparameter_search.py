from ray.tune import schedulers
from ray.tune import suggest
from ray.tune.progress_reporter import CLIReporter
from ray.tune.utils.util import date_str
import torch
from torch.utils.data import DataLoader

from datasets import IronMarch, BertPreprocess
from models import VAE, InterpolatedLinearLayers
from side_information import SideLoader

from torch.optim import Adam

import os

import geomloss

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch

from training import config, VAEBERT, KeepBest

from filelock import FileLock

epsilon = 1e-20
os.environ['TUNE_MAX_LEN_IDENTIFIER'] = '50'
    
def run_optuna_tune():
    config['dataset']['root_directory'] = os.getcwd()
    config['device'] = 'cuda:0'
    algorithm = OptunaSearch()
    algorithm = ConcurrencyLimiter(
        algorithm,
        max_concurrent = 6
    )
    scheduler = ASHAScheduler(
        max_t = 100,
        grace_period = 5,
        reduction_factor = 2
    )
    reporter = CLIReporter(
        metric_columns = ['loss', 'precision', 'recall', 'accuracy', 'f1_score', 'iter'],
        max_error_rows = 5,
        sort_by_metric = True
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
        keep_checkpoints_num = 1,
        checkpoint_score_attr = 'accuracy'
    )

if __name__ == "__main__":
    run_optuna_tune()
