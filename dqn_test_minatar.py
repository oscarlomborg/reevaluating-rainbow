# %%
import itertools
import os
import random
import sys
from collections import deque, namedtuple

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
)

register_envs()

EpisodeStats = namedtuple(
    "EpisodeStats",
    ["episode_lengths", "episode_rewards"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# %%
# Environment
ENV_ID = "MinAtar/Breakout-v1"

env = gym.make(
    ENV_ID,
    render_mode="rgb_array",
)
env.metadata["render_fps"] = 10

state_processor = StateProcessor()

initial_obs = env_reset(env)
processed_obs = state_processor.process(initial_obs)

num_actions = env.action_space.n
input_shape = processed_obs.shape

print("Observation shape (Gym):", initial_obs.shape)
print("Observation shape (PyTorch):", input_shape)
print("Action space:", env.action_space)
print("Number of actions:", num_actions)


# %%
def deep_q_learning(
    env,
    q_estimator,
    target_estimator,
    state_processor,
    num_episodes,
    experiment_dir,
    replay_memory_size=100_000,
    replay_memory_init_size=1_000,
    update_target_estimator_every=1_000,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=100_000,
    batch_size=32,
    record_video_every=None,
):
    """Train a DQN on a MinAtar environment."""

    Transition = namedtuple(
        "Transition",
        ["state", "action", "reward", "next_state", "done"],
    )

    replay_memory = deque(maxlen=replay_memory_size)

    if record_video_every is None:
        record_video_every = max(1, num_episodes // 20)

    estimator_copy = ModelParametersCopier(
        q_estimator,
        target_estimator,
    )

    current_process = psutil.Process()

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    monitor_path = os.path.join(experiment_dir, "monitor")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(monitor_path, exist_ok=True)

    # --------------------------------------------------
    # Training state / checkpoint restore
    # --------------------------------------------------
    total_t = 0
    start_episode = 0

    episode_lengths = np.zeros(num_episodes, dtype=np.int64)
    episode_rewards = np.zeros(num_episodes, dtype=np.float32)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")

        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location=q_estimator.device,
                weights_only=False
            )

            q_estimator.model.load_state_dict(
                ckpt["q_estimator"]
            )
            target_estimator.model.load_state_dict(
                ckpt["target_estimator"]
            )
            q_estimator.optimizer.load_state_dict(
                ckpt["optimizer"]
            )

            q_estimator.global_step = ckpt["global_step"]
            total_t = ckpt["total_t"]
            start_episode = ckpt.get("episode", 0)

            saved_lengths = ckpt.get("episode_lengths")
            saved_rewards = ckpt.get("episode_rewards")

            if saved_lengths is not None:
                n = min(len(saved_lengths), num_episodes)
                episode_lengths[:n] = saved_lengths[:n]

            if saved_rewards is not None:
                n = min(len(saved_rewards), num_episodes)
                episode_rewards[:n] = saved_rewards[:n]

            print(
                f"Resuming from episode {start_episode}, "
                f"env step {total_t}, "
                f"optimizer step {q_estimator.global_step}."
            )

        except (RuntimeError, KeyError) as err:
            raise RuntimeError(
                "Checkpoint is incompatible with the current MinAtar model. "
                "Delete the old checkpoint or use a new experiment directory."
            ) from err

    else:
        estimator_copy.make()

    stats = EpisodeStats(
        episode_lengths=episode_lengths,
        episode_rewards=episode_rewards,
    )

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
    print("Populating replay memory...")

    state = state_processor.process(env_reset(env))

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

        progress_interval = max(1, replay_memory_init_size // 10)

        if (i + 1) % progress_interval == 0:
            print(
                f"\rReplay memory: {i + 1}/{replay_memory_init_size}",
                end="",
            )
            sys.stdout.flush()

    print()

    # --------------------------------------------------
    # Optional video recording
    # --------------------------------------------------
    try:
        from gymnasium.wrappers import RecordVideo

        env = ScaleRender(env)

        env = RecordVideo(
            env,
            video_folder=monitor_path,
            episode_trigger=lambda episode_id: (
                episode_id % record_video_every == 0
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
    for i_episode in range(start_episode, num_episodes):
        state = state_processor.process(env_reset(env))
        loss = None

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

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t + 1

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

            if done:
                break

            state = next_state

        # --------------------------------------------------
        # TensorBoard
        #
        # Use total_t as the x-axis so resumed runs continue
        # instead of restarting from step 0.
        # --------------------------------------------------
        writer = q_estimator.summary_writer

        if writer is not None:
            episode_step = i_episode + 1

            writer.add_scalar(
                "episode/epsilon",
                float(epsilon),
                episode_step,
            )

            writer.add_scalar(
                "episode/reward",
                float(stats.episode_rewards[i_episode]),
                episode_step,
            )

            writer.add_scalar(
                "episode/length",
                int(stats.episode_lengths[i_episode]),
                episode_step,
            )

            writer.add_scalar(
                "system/cpu_usage_percent",
                current_process.cpu_percent(),
                total_t,
            )

            writer.add_scalar(
                "system/v_memory_usage_percent",
                current_process.memory_percent(memtype="vms"),
                total_t,
            )

            writer.flush()

        # --------------------------------------------------
        # Save checkpoint after each completed episode
        # --------------------------------------------------
        torch.save(
            {
                "q_estimator": q_estimator.model.state_dict(),
                "target_estimator": target_estimator.model.state_dict(),
                "optimizer": q_estimator.optimizer.state_dict(),
                "global_step": q_estimator.global_step,
                "total_t": total_t,
                "episode": i_episode + 1,
                "episode_lengths": (
                    stats.episode_lengths[: i_episode + 1].copy()
                ),
                "episode_rewards": (
                    stats.episode_rewards[: i_episode + 1].copy()
                ),
            },
            checkpoint_path,
        )

        if i_episode % record_video_every == 0:
            print(
                f"Episode {i_episode + 1}/{num_episodes} | "
                f"reward={stats.episode_rewards[i_episode]:.1f} | "
                f"length={stats.episode_lengths[i_episode]} | "
                f"epsilon={epsilon:.4f} | "
                f"total_t={total_t} | "
                f"loss={loss:.6f}"
            )

        yield total_t, EpisodeStats(
            episode_lengths=(
                stats.episode_lengths[: i_episode + 1].copy()
            ),
            episode_rewards=(
                stats.episode_rewards[: i_episode + 1].copy()
            ),
        )


# %%
# Experiment setup
experiment_dir = os.path.abspath(
    f"./experiments/{env.spec.id.replace('/', '_')}"
)

q_estimator = Estimator(
    input_shape=input_shape,
    num_actions=num_actions,
    scope="q_estimator",
    summaries_dir=experiment_dir,
    device=device,
)

target_estimator = Estimator(
    input_shape=input_shape,
    num_actions=num_actions,
    scope="target_q",
    device=device,
)

# %%

# Train
num_episodes = 50_000

for t, stats in deep_q_learning(
    env,
    q_estimator=q_estimator,
    target_estimator=target_estimator,
    state_processor=state_processor,
    experiment_dir=experiment_dir,
    num_episodes=num_episodes,
    replay_memory_size=100_000,
    replay_memory_init_size=1_000,
    update_target_estimator_every=1_000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=200_000,
    discount_factor=0.99,
    batch_size=32,
):
    pass

if q_estimator.summary_writer:
    q_estimator.summary_writer.flush()
    q_estimator.summary_writer.close()

env.close()