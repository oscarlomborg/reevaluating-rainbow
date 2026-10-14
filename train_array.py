"""Run all seeds for one environment selected by an LSF job-array index.

Index 1 selects cfg.ENVIRONMENTS[0], index 2 the second environment, etc.
With --seed-procs 2, two seeds run concurrently inside one LSF array job.
"""
import argparse
import multiprocessing as mp
import os
from time import perf_counter

from train import load_config, run_job, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-procs", type=int, default=2,
                        help="Concurrent seed workers within each environment job (1 = sequential)")
    parser.add_argument("--index", type=int, default=None,
                        help="1-based environment index (defaults to LSB_JOBINDEX)")
    args = parser.parse_args()

    if args.seeds < 1 or args.seed_procs < 1:
        parser.error("--seeds and --seed-procs must be positive")

    raw_index = args.index if args.index is not None else os.environ.get("LSB_JOBINDEX")
    if raw_index is None:
        parser.error("Supply --index or submit as an LSF job array")
    try:
        index = int(raw_index)
    except (ValueError, TypeError):
        parser.error(f"Invalid LSF array index: {raw_index!r}")

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    validate_config(cfg)
    if not 1 <= index <= len(cfg.ENVIRONMENTS):
        parser.error(f"Index {index} out of range: 1..{len(cfg.ENVIRONMENTS)}")

    env_id = cfg.ENVIRONMENTS[index - 1]
    jobs = [(env_id, seed, config_path) for seed in range(args.seeds)]
    n_procs = min(args.seed_procs, args.seeds)

    print(f"[ARRAY] index={index}/{len(cfg.ENVIRONMENTS)} "
          f"environment={env_id} seeds={args.seeds} concurrent_seed_workers={n_procs}",
          flush=True)
    start = perf_counter()
    if n_procs == 1:
        results = [run_job(job) for job in jobs]
    else:
        # Explicit spawn, consistent with the original train.py and safer with PyTorch.
        with mp.get_context("spawn").Pool(processes=n_procs) as pool:
            results = pool.map(run_job, jobs, chunksize=1)

    print(f"[ARRAY DONE] environment={env_id} "
          f"completed_seeds={[result['seed'] for result in results]} "
          f"elapsed={perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
