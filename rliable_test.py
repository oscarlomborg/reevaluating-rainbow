# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

# %%
# ============================================================
# Configuration
# ============================================================

EXPERIMENT_ROOT = "./experiments"

GAMES = [
    "MinAtar_Breakout-v1",
    # "MinAtar_Asterix-v1",
    # "MinAtar_Freeway-v1",
    # "MinAtar_Seaquest-v1",
    # "MinAtar_SpaceInvaders-v1",
]

SEEDS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8, 
    9, 
    10,
    11, 
    12,
    13,
    14,
    15
]

ALGORITHM_NAME = "DQN"

# Bootstrap repetitions.
# Use e.g. 2_000 while testing.
# For final figures you can increase to 50_000.
BOOTSTRAP_REPS = 2_000

# %%
# ============================================================
# Loading
# ============================================================

def load_evaluation_file(game, seed):
    path = os.path.join(
        EXPERIMENT_ROOT,
        game,
        f"seed_{seed}",
        "evaluation.npz",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find evaluation file:\n{path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    )

    frames = np.asarray(
        data["frames"],
        dtype=np.int64,
    )

    mean_returns = np.asarray(
        data["mean_returns"],
        dtype=np.float64,
    )

    return frames, mean_returns


def load_all_scores():
    """
    Returns
    -------
    frames:
        shape (num_eval_points,)

    scores:
        shape (
            num_seeds,
            num_games,
            num_eval_points
        )
    """

    all_game_scores = []
    reference_frames = None

    for game in GAMES:

        seed_scores = []

        for seed in SEEDS:
            frames, returns = load_evaluation_file(
                game,
                seed,
            )

            if reference_frames is None:
                reference_frames = frames
            else:
                if not np.array_equal(
                    frames,
                    reference_frames,
                ):
                    raise ValueError(
                        f"Evaluation frames do not match "
                        f"for game={game}, seed={seed}.\n"
                        f"Expected: {reference_frames}\n"
                        f"Got:      {frames}"
                    )

            seed_scores.append(returns)

        # (num_seeds, num_eval_points)
        seed_scores = np.stack(
            seed_scores,
            axis=0,
        )

        all_game_scores.append(seed_scores)

    # Current shape:
    # (num_games, num_seeds, num_eval_points)
    scores = np.stack(
        all_game_scores,
        axis=0,
    )

    # Convert to rliable layout:
    #
    # (num_seeds, num_games, num_eval_points)
    scores = np.transpose(
        scores,
        (1, 0, 2),
    )

    return reference_frames, scores

# %%
# ============================================================
# Load data
# ============================================================

frames, scores = load_all_scores()

print("Frames:")
print(frames)

print("\nScore tensor shape:")
print(scores.shape)

print(
    "\nExpected layout:"
    "\n(num_seeds, num_games, num_eval_points)"
)

print(
    f"\nSeeds: {len(SEEDS)}"
    f"\nGames: {len(GAMES)}"
    f"\nEvaluation points: {len(frames)}"
)

# %%
# ============================================================
# Rliable dictionaries
# ============================================================

score_dict = {
    ALGORITHM_NAME: scores
}

# Final evaluation point:
#
# (num_seeds, num_games)
final_scores = scores[:, :, -1]

final_score_dict = {
    ALGORITHM_NAME: final_scores
}

print("\nFinal-score shape:")
print(final_scores.shape)

# %%
# ============================================================
# 1. Aggregate metrics
#
# Mean
# Median
# IQM
# ============================================================

def aggregate_metrics(x):
    return np.array([
        metrics.aggregate_mean(x),
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
    ])


aggregate_scores, aggregate_cis = (
    rly.get_interval_estimates(
        final_score_dict,
        aggregate_metrics,
        reps=BOOTSTRAP_REPS,
    )
)

print("\n========================================")
print("Aggregate metrics")
print("========================================")

print(
    "Mean, Median, IQM:"
)

print(
    aggregate_scores[
        ALGORITHM_NAME
    ]
)

print("\n95% confidence intervals:")

print(
    aggregate_cis[
        ALGORITHM_NAME
    ]
)

# %%
# ============================================================
# Plot aggregate metrics
# ============================================================

fig, ax = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_cis,
    metric_names=[
        "Mean",
        "Median",
        "IQM",
    ],
    algorithms=[
        ALGORITHM_NAME,
    ],
    xlabel="Evaluation return",
)

plt.tight_layout()
plt.show()

# %%
# ============================================================
# 2. IQM sample-efficiency curve
#
# Input shape:
# (seed, game, evaluation point)
# ============================================================

def iqm_over_time(x):
    return np.array([
        metrics.aggregate_iqm(
            x[..., i]
        )
        for i in range(
            x.shape[-1]
        )
    ])


iqm_scores, iqm_cis = (
    rly.get_interval_estimates(
        score_dict,
        iqm_over_time,
        reps=BOOTSTRAP_REPS,
    )
)

print("\n========================================")
print("IQM learning curve")
print("========================================")

print(
    iqm_scores[
        ALGORITHM_NAME
    ]
)


plot_utils.plot_sample_efficiency_curve(
    frames,
    iqm_scores,
    iqm_cis,
    algorithms=[
        ALGORITHM_NAME,
    ],
    xlabel="Training frames",
    ylabel="IQM evaluation return",
)

plt.tight_layout()
plt.show()

# %%
# ============================================================
# 3. Performance profile
#
# For raw MinAtar returns, choose thresholds based on
# the observed score range.
#
# For a paper, normalized scores are often preferable
# when aggregating different games.
# ============================================================

minimum_score = np.min(
    final_scores
)

maximum_score = np.max(
    final_scores
)

thresholds = np.linspace(
    minimum_score,
    maximum_score,
    100,
)

performance_profiles, performance_profile_cis = (
    rly.create_performance_profile(
        final_score_dict,
        thresholds,
        reps=BOOTSTRAP_REPS,
    )
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

plot_utils.plot_performance_profiles(
    performance_profiles,
    thresholds,
    performance_profile_cis=(
        performance_profile_cis
    ),
    xlabel="Evaluation return threshold",
    ax=ax,
)

plt.tight_layout()
plt.show()

# %%
# ============================================================
# 4. Raw per-seed final scores
# ============================================================

print("\n========================================")
print("Final scores per seed/game")
print("========================================")

for seed_index, seed in enumerate(SEEDS):

    print(
        f"\nSeed {seed}:"
    )

    for game_index, game in enumerate(GAMES):

        score = final_scores[
            seed_index,
            game_index,
        ]

        print(
            f"  {game}: {score:.3f}"
        )

# %%
# ============================================================
# 5. Raw mean/std summary
#
# Not specifically a rliable metric,
# but useful as a sanity check.
# ============================================================

print("\n========================================")
print("Per-game mean ± std")
print("========================================")

for game_index, game in enumerate(GAMES):

    game_scores = final_scores[
        :,
        game_index,
    ]

    mean = np.mean(game_scores)
    std = np.std(
        game_scores,
        ddof=1,
    )

    print(
        f"{game}: "
        f"{mean:.3f} ± {std:.3f}"
    )

# %%
# ============================================================
# 6. Optionality gap demonstration
#
# IMPORTANT:
# The default threshold gamma=1 only makes sense when
# scores have been normalized such that 1 corresponds
# to your reference/optimal score.
#
# Therefore this is NOT automatically appropriate for
# raw MinAtar returns.
# ============================================================

USE_NORMALIZED_SCORES = False

if USE_NORMALIZED_SCORES:

    def normalized_aggregate_metrics(x):
        return np.array([
            metrics.aggregate_iqm(x),
            metrics.aggregate_optimality_gap(
                x,
                gamma=1.0,
            ),
        ])

    normalized_scores, normalized_cis = (
        rly.get_interval_estimates(
            final_score_dict,
            normalized_aggregate_metrics,
            reps=BOOTSTRAP_REPS,
        )
    )

    print("\nIQM / Optimality gap:")
    print(
        normalized_scores[
            ALGORITHM_NAME
        ]
    )


print("\nDone.")