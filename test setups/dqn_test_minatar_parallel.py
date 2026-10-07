# %%
import itertools
import multiprocessing as mp
import os
import random
import sys
from collections import deque, namedtuple
from time import perf_counter

# Limit native numerical libraries to one thread per worker.
# These must be set before importing NumPy and PyTorch.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import gymnasium as gym
import numpy as np
import psutil
import torch
from minatar.gym import register_envs

from setup_minatar import (
    Estimator,
    ModelParametersCopier,
    ScaleRender,
    StateProcessor,
    env_reset,
    env_step,
    make_epsilon_greedy_policy,
    evaluate_policy
)

register_envs()

EpisodeStats = namedtuple(
    "EpisodeStats",
    ["episode_lengths", "episode_rewards"],
)

# Edit this list to choose the environments to run.
ENVIRONMENTS = [
    "MinAtar/Breakout-v1"
    # "MinAtar/Asterix-v1",
    # "MinAtar/Seaquest-v1",
    # "MinAtar/SpaceInvaders-v1",
    # "MinAtar/Freeway-v1",
]

NUM_FRAMES = 500_000
RESUME_FROM_CHECKPOINT = False


def deep_q_learning(
    env,
    eval_env,
    env_id,
    q_estimator,
    target_estimator,
    state_processor,
    num_frames,
    seed_dir,
    seed=0,
    eval_every=50_000,
    eval_len=10_000,
    replay_memory_size=100_000,
    replay_memory_init_size=1_000,
    update_target_estimator_every=1_000,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=100_000,
    batch_size=32,
    record_video_every=None,
    resume_from_checkpoint=False,
):
    """Train a DQN on one environment and seed."""

    checkpoint_every = max(1, num_frames // 100)
    print_every = max(1, num_frames // 20)


    Transition = namedtuple(
        "Transition",
        ["state", "action", "reward", "next_state", "done"],
    )

    replay_memory = deque(maxlen=replay_memory_size)

    estimator_copy = ModelParametersCopier(
        q_estimator,
        target_estimator,
    )

    current_process = psutil.Process()

    checkpoint_dir = os.path.join(seed_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    monitor_path = os.path.join(seed_dir, "monitor")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(monitor_path, exist_ok=True)

    # --------------------------------------------------
    # Training state / checkpoint restore
    # --------------------------------------------------
    total_t = 0
    start_episode = 0

    episode_lengths = []
    episode_rewards = []

    evaluation_frames = []
    evaluation_returns = []
    evaluation_mean_returns = []

    next_checkpoint_t = (
        (total_t // checkpoint_every) + 1
    ) * checkpoint_every

    next_eval_t = (
        (total_t // eval_every) + 1
    ) * eval_every

    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")

        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location=q_estimator.device,
                weights_only=False,
            )

            # Restore networks and optimizer
            q_estimator.model.load_state_dict(
                ckpt["q_estimator"]
            )

            target_estimator.model.load_state_dict(
                ckpt["target_estimator"]
            )

            q_estimator.optimizer.load_state_dict(
                ckpt["optimizer"]
            )

            # Restore training state
            q_estimator.global_step = ckpt["global_step"]
            total_t = ckpt["total_t"]
            start_episode = ckpt.get("episode", 0)

            # Restore completed episode statistics
            saved_lengths = ckpt.get("episode_lengths")
            saved_rewards = ckpt.get("episode_rewards")

            if saved_lengths is not None:
                episode_lengths.extend(
                    np.asarray(saved_lengths).tolist()
                )

            if saved_rewards is not None:
                episode_rewards.extend(
                    np.asarray(saved_rewards).tolist()
                )

            print(
                f"Resuming from episode {start_episode}, "
                f"frame {total_t}/{num_frames}, "
                f"optimizer step {q_estimator.global_step}."
            )

        except (RuntimeError, KeyError) as err:
            raise RuntimeError(
                "Checkpoint is incompatible with the current MinAtar model. "
                "Delete the old checkpoint or use a new experiment directory."
            ) from err

    else:
        estimator_copy.make()

    # --------------------------------------------------
    # Epsilon schedule / policy
    # --------------------------------------------------
    epsilons = np.linspace(
        epsilon_start,
        epsilon_end,
        epsilon_decay_steps,
        dtype=np.float32,
    )

    num_actions = env.action_space.n

    policy = make_epsilon_greedy_policy(
        q_estimator,
        num_actions,
    )

    # --------------------------------------------------
    # Populate replay memory
    # --------------------------------------------------

    state = state_processor.process(
        env_reset(env, seed=seed)
    )

    for i in range(replay_memory_init_size):
        epsilon = epsilons[
            min(total_t, epsilon_decay_steps - 1)
        ]

        action_probs = policy(state, epsilon)
        action = np.random.choice(
            num_actions,
            p=action_probs,
        )

        next_state, reward, done, _ = env_step(
            env,
            action,
        )

        next_state = state_processor.process(next_state)

        replay_memory.append(
            Transition(
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

        total_t += 1

        if done:
            state = state_processor.process(env_reset(env))
        else:
            state = next_state

    # --------------------------------------------------
    # Optional video recording
    # --------------------------------------------------
    if record_video_every is not None:
        try:
            from gymnasium.wrappers import RecordVideo

            env = ScaleRender(env)

            env = RecordVideo(
                env,
                video_folder=monitor_path,
                step_trigger=lambda step_id: (
                    step_id % record_video_every == 0
                ),
            )

        except (ImportError, ValueError) as err:
            print(
                f"Video recording disabled ({err}); "
                "continuing without it."
            )
    # --------------------------------------------------
    # Main training loop
    # --------------------------------------------------
    i_episode = start_episode

    while total_t < num_frames:
        state = state_processor.process(env_reset(env))
        loss = None

        episode_reward = 0.0
        episode_length = 0

        for t in itertools.count():
            epsilon = epsilons[
                min(total_t, epsilon_decay_steps - 1)
            ]

            # Periodically synchronize target network.
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make()

            # Epsilon-greedy action.
            action_probs = policy(state, epsilon)

            action = np.random.choice(
                num_actions,
                p=action_probs,
            )

            next_state, reward, done, _ = env_step(
                env,
                action,
            )

            next_state = state_processor.process(next_state)

            replay_memory.append(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            )

            episode_reward += reward
            episode_length += 1

            # ------------------------------------------
            # Sample minibatch
            # ------------------------------------------
            samples = random.sample(
                replay_memory,
                batch_size,
            )

            (
                states_batch,
                action_batch,
                reward_batch,
                next_states_batch,
                done_batch,
            ) = map(np.array, zip(*samples))

            # ------------------------------------------
            # DQN target
            # ------------------------------------------
            q_values_next = target_estimator.predict(
                next_states_batch
            )

            not_done = (
                1.0
                - done_batch.astype(np.float32)
            )

            targets_batch = (
                reward_batch
                + not_done
                * discount_factor
                * np.max(q_values_next, axis=1)
            )

            # ------------------------------------------
            # Gradient update
            # ------------------------------------------
            loss = q_estimator.update(
                states_batch,
                action_batch,
                targets_batch,
            )

            total_t += 1

            # ------------------------------------------
            # validation
            # ------------------------------------------

            if total_t >= next_eval_t:

                eval_results = evaluate_policy(
                    env=eval_env,
                    q_estimator=q_estimator,
                    state_processor=state_processor,
                    eval_len=eval_len,
                    seed=seed + 1,
                )

                eval_rewards = eval_results["rewards"]
                eval_lengths = eval_results["lengths"]

                evaluation_frames.append(total_t)
                evaluation_returns.append(eval_rewards.copy())
                evaluation_mean_returns.append(float(np.mean(eval_rewards)))

                np.savez(
                    os.path.join(seed_dir, "evaluation.npz"),
                    frames=np.asarray(evaluation_frames, dtype=np.int64),
                    mean_returns=np.asarray(evaluation_mean_returns, dtype=np.float32),
                    raw_returns=np.asarray(evaluation_returns, dtype=object),
                    seed=seed,
                    game=env_id,
                )

                if len(eval_rewards) > 0:
                    writer = q_estimator.summary_writer

                    if writer is not None:
                        writer.add_scalar(
                            "eval/return_mean",
                            float(np.mean(eval_rewards)),
                            total_t,
                        )

                        writer.add_scalar(
                            "eval/return_median",
                            float(np.median(eval_rewards)),
                            total_t,
                        )

                        writer.add_scalar(
                            "eval/episode_length_mean",
                            float(np.mean(eval_lengths)),
                            total_t,
                        )

                        writer.add_scalar(
                            "eval/num_episodes",
                            len(eval_rewards),
                            total_t,
                        )

                        writer.flush()

                next_eval_t = (
                    (total_t // eval_every) + 1
                ) * eval_every

            if done or total_t >= num_frames:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        i_episode += 1

        # --------------------------------------------------
        # TensorBoard
        #
        # Use total_t as the x-axis so resumed runs continue
        # instead of restarting from step 0.
        # --------------------------------------------------
        writer = q_estimator.summary_writer

        if writer is not None:
            writer.add_scalar(
                "episode/epsilon",
                float(epsilon),
                i_episode,
            )

            writer.add_scalar(
                "episode/reward",
                float(episode_reward),
                i_episode,
            )

            writer.add_scalar(
                "episode/length",
                int(episode_length),
                i_episode,
            )

            writer.add_scalar(
                "system/cpu_usage_percent",
                current_process.cpu_percent(),
                total_t,
            )

            mem = current_process.memory_info()

            writer.add_scalar(
                "system/memory_rss_mb",
                mem.rss / (1024 ** 2),
                total_t,
            )

            writer.add_scalar(
                "system/memory_vms_mb",
                mem.vms / (1024 ** 2),
                total_t,
            )

            writer.flush()

        # --------------------------------------------------
        # Save checkpoints
        # --------------------------------------------------
        if total_t >= next_checkpoint_t or total_t == num_frames:
            torch.save(
                {
                    "q_estimator": q_estimator.model.state_dict(),
                    "target_estimator": target_estimator.model.state_dict(),
                    "optimizer": q_estimator.optimizer.state_dict(),
                    "global_step": q_estimator.global_step,
                    "total_t": total_t,
                    "episode": i_episode,
                    "episode_lengths": np.array(
                        episode_lengths,
                        dtype=np.int64,
                    ),
                    "episode_rewards": np.array(
                        episode_rewards,
                        dtype=np.float32,
                    ),
                },
                checkpoint_path,
            )

            next_checkpoint_t = (
                (total_t // checkpoint_every) + 1
            ) * checkpoint_every

    stats = EpisodeStats(
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        episode_rewards=np.asarray(episode_rewards, dtype=np.float32),
    )
    np.savez(
        os.path.join(seed_dir, "training.npz"),
        episode_lengths=stats.episode_lengths,
        episode_rewards=stats.episode_rewards,
        seed=seed,
        game=env_id,
    )
    return stats


def run_job(job):
    """Run an independent (environment ID, seed) experiment."""
    env_id, seed = job
    torch.set_num_threads(1)
    device = torch.device("cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game_name = env_id.replace("/", "_")
    seed_dir = os.path.abspath(
        os.path.join("experiments", game_name, f"seed_{seed}")
    )
    os.makedirs(seed_dir, exist_ok=True)

    env = None
    eval_env = None
    q_estimator = None
    target_estimator = None
    start = perf_counter()

    print(
        f"[START] env={env_id}, seed={seed}, "
        f"pid={os.getpid()}, device={device}",
        flush=True,
    )

    try:
        env = gym.make(env_id, render_mode="rgb_array")
        eval_env = gym.make(env_id, render_mode="rgb_array")
        env.metadata["render_fps"] = 10
        eval_env.metadata["render_fps"] = 10

        for current_env in (env, eval_env):
            current_env.action_space.seed(seed)
            current_env.observation_space.seed(seed)

        state_processor = StateProcessor()
        initial_obs = env_reset(env, seed=seed)
        input_shape = state_processor.process(initial_obs).shape
        num_actions = env.action_space.n

        q_estimator = Estimator(
            input_shape=input_shape,
            num_actions=num_actions,
            scope="q_estimator",
            summaries_dir=seed_dir,
            device=device,
        )
        target_estimator = Estimator(
            input_shape=input_shape,
            num_actions=num_actions,
            scope="target_q",
            device=device,
        )

        stats = deep_q_learning(
            env=env,
            eval_env=eval_env,
            env_id=env_id,
            q_estimator=q_estimator,
            target_estimator=target_estimator,
            state_processor=state_processor,
            num_frames=NUM_FRAMES,
            seed_dir=seed_dir,
            seed=seed,
            eval_every=50_000,
            eval_len=10_000,
            replay_memory_size=100_000,
            replay_memory_init_size=1_000,
            update_target_estimator_every=1_000,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=200_000,
            discount_factor=0.99,
            batch_size=32,
            resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        )

        elapsed = perf_counter() - start
        print(
            f"[DONE] env={env_id}, seed={seed}, "
            f"episodes={len(stats.episode_rewards)}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

        # Statistics are saved by the worker; return only a small summary.
        return {
            "env_id": env_id,
            "seed": seed,
            "seed_dir": seed_dir,
            "episodes": len(stats.episode_rewards),
            "elapsed_seconds": elapsed,
        }
    finally:
        try:
            for estimator in (q_estimator, target_estimator):
                writer = getattr(estimator, "summary_writer", None)
                if writer is not None:
                    try:
                        writer.flush()
                    finally:
                        writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if eval_env is not None:
                    eval_env.close()


if __name__ == "__main__":
    max_procs = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    NUMBER_OF_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    if max_procs < 1 or NUMBER_OF_SEEDS < 1:
        raise ValueError("max_procs and NUMBER_OF_SEEDS must be at least 1")

    seeds = list(range(NUMBER_OF_SEEDS))
    jobs = list(itertools.product(ENVIRONMENTS, seeds))
    n_procs = min(max_procs, len(jobs))

    print(f"--- Running {len(ENVIRONMENTS)} environments x {NUMBER_OF_SEEDS} seeds with {n_procs} processes ---", flush=True)

    start = perf_counter()

    # Start fresh worker processes for PyTorch.
    mp.set_start_method("spawn")
    with mp.Pool(n_procs) as pool:
        results = pool.map(run_job, jobs, chunksize=1)

    elapsed = perf_counter() - start
    print(f"--- Finished {len(jobs)} runs in {elapsed:.1f} seconds ---", flush=True)
