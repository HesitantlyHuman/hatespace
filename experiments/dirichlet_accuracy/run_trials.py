from typing import Iterable, List, Any, Tuple
import hatespace
from tqdm import tqdm
import torch
import geomloss
import csv
import os

# TODO add a cli interface

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"Using {DEVICE}...")

loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05)


def run_trial(
    sampler: torch.distributions.Distribution, num_samples: int, batch_size: int
) -> torch.Tensor:
    batch = sampler.sample([batch_size]).to(DEVICE)
    samples = sampler.sample([num_samples]).to(DEVICE)
    return loss_fn(batch, samples)


def run_experiment(
    alpha: float,
    dimensions: int,
    num_samples: int,
    batch_size: int,
    trials_to_run: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    results = []
    sampler = torch.distributions.Dirichlet(
        torch.tensor([alpha for _ in range(dimensions)])
    )
    for _ in tqdm(range(trials_to_run), desc="Trials", leave=False):
        results.append(run_trial(sampler, num_samples, batch_size))
    return results


def log_space(start, stop, steps) -> List[int]:
    start = torch.log(torch.tensor(start)) / torch.log(torch.tensor(10))
    stop = torch.log(torch.tensor(stop)) / torch.log(torch.tensor(10))
    num_samples = list(set(torch.logspace(start, stop, steps).round().long().tolist()))
    num_samples.sort()
    num_samples.reverse()
    return num_samples


def value_tqdm(iterator: Iterable, leave: bool = True, desc: str = "Unknown") -> Any:
    pbar = tqdm(iterator, leave=leave, smoothing=1.0)
    for element in pbar:
        pbar.set_description(f"{desc} {element}".ljust(25))
        yield element


def create_csv_with_headers(path: str, headers: List[str]) -> None:
    with open(path, "w+") as csv_file:
        print("Creating csv...")
        csv_file.write(",".join(headers) + "\n")
        csv_file.flush()


csv_path = "experiments/dirichlet_accuracy/trial_data.csv"

if not os.path.exists(csv_path):
    create_csv_with_headers(
        csv_path,
        [
            "dimensions",
            "alpha",
            "num_sampled",
            "batch_size",
            "trials",
            "error",
        ],
    )

TRIALS_PER_EXPERIMENT = 50

latent_dimensions = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64]
alphas = [0.05, 0.1, 0.5, 1.0, 5.0]
num_samples = log_space(2, 32000, 15)
batch_sizes = [2, 3, 4, 6, 8, 16, 32]

total_trials = (
    len(latent_dimensions) * len(alphas) * len(num_samples) * len(batch_sizes)
)
total_pbar = tqdm(total=total_trials, desc=f"Total Progress", position=0)
with open(csv_path, "a") as csv_file:
    writer = csv.writer(csv_file)
    for dimensions in value_tqdm(latent_dimensions, leave=False, desc="Dimensions"):
        for alpha in value_tqdm(alphas, leave=False, desc="Alpha"):
            for num_to_sample in value_tqdm(
                num_samples, leave=False, desc="Num Samples"
            ):
                for batch_size in value_tqdm(
                    batch_sizes, leave=False, desc="Batch Sizes"
                ):
                    errors = run_experiment(
                        alpha,
                        dimensions,
                        num_to_sample,
                        batch_size,
                        trials_to_run=TRIALS_PER_EXPERIMENT,
                    )
                    for error in errors:
                        error = error.item()
                        writer.writerow(
                            [
                                dimensions,
                                alpha,
                                num_to_sample,
                                batch_size,
                                TRIALS_PER_EXPERIMENT,  # TODO Remove
                                error,
                            ]
                        )
                    total_pbar.update(1)
