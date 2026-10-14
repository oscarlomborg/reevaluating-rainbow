# %%
import argparse
import importlib.util
import itertools
import multiprocessing as mp
import os
import random
import tracemalloc
from collections import deque, namedtuple
from time import perf_counter

# TODO memory
import gc
import ctypes

# One numerical thread per worker.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import gymnasium as gym
import numpy as np
import psutil
import torch

try:
    from minatar.gym import register_envs
except ImportError:
    register_envs = None

from SETUP import (
    Estimator,
    ModelParametersCopier,
    StateProcessor,
    build_c51_targets,
    build_dqn_targets,
    env_reset,
    env_step,
    evaluate_policy,
    make_epsilon_greedy_policy,
)


###################################################
# Config helpers
###################################################

REQUIRED_CONFIG = (
    "ENV_TYPE", "NAME", "DOUBLE_DQN", "N_STEP", "PRIORITIZED_REPLAY",
    "NETWORK_TYPE", "NOISY", "NUM_ATOMS", "ENVIRONMENTS",
    "NUM_FRAMES", "REPLAY_PERIOD",
    "REPLAY_MEMORY_SIZE", "REPLAY_MEMORY_INIT_SIZE",
    "PER_ALPHA", "PER_BETA_START", "PER_BETA_FRAMES", "PER_EPS",
    "DISCOUNT_FACTOR", "BATCH_SIZE", "TARGET_UPDATE_EVERY",
    "EPSILON_START", "EPSILON_END", "EPSILON_DECAY_STEPS",
    "EVAL_EVERY", "EVAL_LEN", "NOOP_MAX",
    "ADAM_LR", "ADAM_EPS", "V_MIN", "V_MAX"
)


def load_config(path):
    """Load a Python config file as a module."""
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config: {path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def validate_config(cfg):
    """Fail early if a required setting is missing or invalid."""
    missing = [name for name in REQUIRED_CONFIG if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f"Config is missing: {', '.join(missing)}")

    if cfg.ENV_TYPE not in ("minatar", "atari"):
        raise ValueError("ENV_TYPE must be 'minatar' or 'atari'.")
    if cfg.NETWORK_TYPE not in ("dqn", "dueling", "distributional", "distributional_dueling"):
        raise ValueError(f"Unknown NETWORK_TYPE: {cfg.NETWORK_TYPE}")
    if cfg.N_STEP < 1:
        raise ValueError("N_STEP must be at least 1.")
    if cfg.REPLAY_PERIOD < 1:
        raise ValueError("REPLAY_PERIOD must be at least 1.")
    if cfg.TARGET_UPDATE_EVERY < 1 or cfg.EVAL_EVERY < 1:
        raise ValueError("TARGET_UPDATE_EVERY and EVAL_EVERY must be at least 1.")
    if cfg.EPSILON_DECAY_STEPS < 1:
        raise ValueError("EPSILON_DECAY_STEPS must be at least 1.")
    if cfg.NOOP_MAX < 0:
        raise ValueError("NOOP_MAX cannot be negative.")

    # Adam settings.
    if cfg.ADAM_LR <= 0:
        raise ValueError("ADAM_LR must be positive.")
    if cfg.ADAM_EPS <= 0:
        raise ValueError("ADAM_EPS must be positive.")

    # C51 requires an ordered support with at least two atoms.
    if "distributional" in cfg.NETWORK_TYPE:
        if cfg.NUM_ATOMS < 2:
            raise ValueError("NUM_ATOMS must be at least 2 for C51.")
        if cfg.V_MIN >= cfg.V_MAX:
            raise ValueError("V_MIN must be smaller than V_MAX.")
    if cfg.BATCH_SIZE < 1 or cfg.REPLAY_MEMORY_INIT_SIZE < cfg.BATCH_SIZE:
        raise ValueError("REPLAY_MEMORY_INIT_SIZE must be >= BATCH_SIZE >= 1.")
    if cfg.REPLAY_MEMORY_SIZE < cfg.REPLAY_MEMORY_INIT_SIZE:
        raise ValueError("REPLAY_MEMORY_SIZE must be >= REPLAY_MEMORY_INIT_SIZE.")
    if not cfg.ENVIRONMENTS:
        raise ValueError("ENVIRONMENTS cannot be empty.")


def register_environment_type(cfg):
    """Register MinAtar only when the selected config needs it."""
    if cfg.ENV_TYPE == "minatar":
        if register_envs is None:
            raise ImportError("ENV_TYPE='minatar' requires the minatar package.")
        register_envs()


def config_name(cfg):
    """Unique directory name for the selected ablation."""
    return f"{cfg.ENV_TYPE}_{cfg.NAME}"


###################################################
# Small helpers
###################################################

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])


def make_env(env_id, env_type, seed, noop_max=0):
    """Create a MinAtar or preprocessed Atari environment."""
    if env_type == "minatar":
        env = gym.make(env_id)
    else:
        # Atari preprocessing also handles Atari no-op starts.
        env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            scale_obs=False,
        )
        # Gymnasium renamed FrameStack -> FrameStackObservation in newer versions.
        if hasattr(gym.wrappers, "FrameStackObservation"):
            env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        else:
            env = gym.wrappers.FrameStack(env, num_stack=4)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


###################################################
# Replay buffer: uniform or prioritized
###################################################

class ReplayBuffer:
    def __init__(self, capacity, prioritized=False, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.eps = eps
        self.data, self.pos = [], 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def add(self, transition):
        # New PER samples receive the current maximum priority.
        priority = self.priorities[:len(self.data)].max() if self.data else 1.0
        priority = max(float(priority), self.eps)

        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=1.0):
        n = len(self.data)
        if self.prioritized:
            probs = self.priorities[:n] ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(n, batch_size, p=probs)
            weights = (n * probs[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = np.random.choice(n, batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)

        return [self.data[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        if self.prioritized:
            self.priorities[indices] = np.maximum(np.asarray(priorities), self.eps)


###################################################
# N-step transition construction
###################################################

def _make_n_step(buffer, n_step, gamma):
    """Collapse the oldest prefix into one n-step transition."""
    first = buffer[0]
    reward, next_state, done = 0.0, first.next_state, False

    for i, tr in enumerate(list(buffer)[:n_step]):
        reward += (gamma ** i) * tr.reward
        next_state, done = tr.next_state, tr.done
        if done:
            break

    return Transition(first.state, first.action, reward, next_state, done)


def add_n_step(buffer, replay, transition, n_step, gamma):
    """Add a raw transition and emit complete n-step transitions."""
    buffer.append(transition)

    if len(buffer) >= n_step:
        replay.add(_make_n_step(buffer, n_step, gamma))
        buffer.popleft()

    # Flush shortened terminal prefixes; done=True means no bootstrap term.
    if transition.done:
        while buffer:
            replay.add(_make_n_step(buffer, n_step, gamma))
            buffer.popleft()


###################################################
# PER priority signal
###################################################

def replay_priorities(estimator, states, actions, targets, eps):
    """TD error for scalar DQN; per-sample cross entropy for C51."""
    idx = np.arange(len(actions))

    if estimator.distributional:
        pred = estimator.predict_dist(states)[idx, actions]
        return -(targets * np.log(np.clip(pred, 1e-8, 1.0))).sum(axis=1) + eps

    pred = estimator.predict(states)[idx, actions]
    return np.abs(targets - pred) + eps


###################################################
# Training
###################################################

def train_model(
    env, eval_env, env_id, q_estimator, target_estimator,
    state_processor, cfg, seed_dir, seed=0,
    mem_profile=False, mem_every=5000, stop_after=None, no_updates=False,
):
    """Train one configurable DQN variant on one environment and seed."""

    checkpoint_every = max(1, cfg.NUM_FRAMES // 100)
    # Debug-only: stop early WITHOUT changing checkpoint/evaluation schedules.
    run_until = min(cfg.NUM_FRAMES, stop_after) if stop_after else cfg.NUM_FRAMES
    replay = ReplayBuffer(
        cfg.REPLAY_MEMORY_SIZE,
        prioritized=cfg.PRIORITIZED_REPLAY,
        alpha=cfg.PER_ALPHA,
        eps=cfg.PER_EPS,
    )
    nstep_buffer = deque()
    copier = ModelParametersCopier(q_estimator, target_estimator)
    current_process = psutil.Process()
    current_process.cpu_percent(None)  # Prime CPU measurement.

    checkpoint_dir = os.path.join(seed_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_t = 0
    episode_lengths, episode_rewards = [], []
    evaluation_frames, evaluation_returns, evaluation_mean_returns = [], [], []

    def report_memory(phase):
        if not mem_profile:
            return
        info = current_process.memory_info()
        try:
            uss_mb = current_process.memory_full_info().uss / (1024**2)
        except (AttributeError, OSError):
            uss_mb = float('nan')
        heap_mb, peak_mb = tracemalloc.get_traced_memory()
        # The numpy observation bytes retained in replay, excluding Python overhead.
        replay_data_mb = sum(
            t.state.nbytes + t.next_state.nbytes for t in replay.data
        ) / (1024**2)
        shape = replay.data[0].state.shape if replay.data else None
        dtype = replay.data[0].state.dtype if replay.data else None
        print(
            f"[MEM] {phase} env={env_id} seed={seed} steps={total_t} "
            f"RSS={info.rss/1024**2:.1f}MiB "
            f"USS={uss_mb:.1f}MiB "
            f"PythonTracked={heap_mb/1024**2:.1f}MiB "
            f"PythonPeak={peak_mb/1024**2:.1f}MiB "
            f"ReplayBytes={replay_data_mb:.1f}MiB "
            f"ReplayLen={len(replay)} shape={shape} dtype={dtype} "
            f"threads={current_process.num_threads()}",
            flush=True,
        )

    report_memory('initialized')

    # Start every run from a fresh model.
    copier.make()

    next_checkpoint_t = ((total_t // checkpoint_every) + 1) * checkpoint_every
    next_eval_t = ((total_t // cfg.EVAL_EVERY) + 1) * cfg.EVAL_EVERY
    epsilons = np.linspace(
        cfg.EPSILON_START, cfg.EPSILON_END, cfg.EPSILON_DECAY_STEPS, dtype=np.float32
    )
    policy = make_epsilon_greedy_policy(
        q_estimator, env.action_space.n, use_noisy_exploration=cfg.NOISY
    )

    # MinAtar uses our reset helper for no-op starts.
    # AtariPreprocessing already performs Atari no-op starts.
    reset_noop_max = cfg.NOOP_MAX if cfg.ENV_TYPE == "minatar" else 0

    # --------------------------------------------------
    # Populate replay memory.
    # --------------------------------------------------
    state = state_processor.process(
        env_reset(env, seed=seed, noop_max=reset_noop_max)
    )
    while len(replay) < cfg.REPLAY_MEMORY_INIT_SIZE and total_t < cfg.NUM_FRAMES:
        # NoisyNet replaces epsilon-greedy exploration.
        epsilon = 0.0 if cfg.NOISY else 1.0
        action = policy(state, epsilon)
        next_state, reward, done, _ = env_step(env, action)
        next_state = state_processor.process(next_state)

        add_n_step(
            nstep_buffer, replay,
            Transition(state, action, reward, next_state, done),
            cfg.N_STEP, cfg.DISCOUNT_FACTOR,
        )
        total_t += 1
        # Seed only the first reset; later resets continue the RNG sequence.
        state = (
            state_processor.process(env_reset(env, noop_max=reset_noop_max))
            if done else next_state
        )

    # Discard an unfinished warm-up prefix before measured episodes.
    nstep_buffer.clear()

    # Replay memory is now populated.
    # Epsilon decay starts from this point.
    learning_start_t = total_t

    # --------------------------------------------------
    # Main loop.
    # --------------------------------------------------
    i_episode = 0
    beta = cfg.PER_BETA_START  # Safe initial value for logging.

    while total_t < run_until:
        # Start a new episode without reseeding the environment.
        state = state_processor.process(env_reset(env, noop_max=reset_noop_max))
        nstep_buffer.clear()
        episode_reward = episode_length = 0

        for _ in itertools.count():

            # Number of environment steps since learning started.
            training_t = total_t - learning_start_t

            # NoisyNet exploration uses epsilon = 0.
            epsilon = 0.0 if cfg.NOISY else epsilons[min(training_t, cfg.EPSILON_DECAY_STEPS - 1)]

            if total_t % cfg.TARGET_UPDATE_EVERY == 0:
                copier.make()

            action = policy(state, epsilon)
            next_state, reward, done, _ = env_step(env, action)
            next_state = state_processor.process(next_state)

            add_n_step(
                nstep_buffer, replay,
                Transition(state, action, reward, next_state, done),
                cfg.N_STEP, cfg.DISCOUNT_FACTOR,
            )

            episode_reward += reward
            episode_length += 1

            if not no_updates and total_t % cfg.REPLAY_PERIOD == 0:

                # Sample uniformly or with PER.
                beta = min(
                    1.0,
                    cfg.PER_BETA_START
                    + total_t * (1.0 - cfg.PER_BETA_START) / max(1, cfg.PER_BETA_FRAMES),
                )
                samples, indices, weights = replay.sample(cfg.BATCH_SIZE, beta)
                states_b, actions_b, rewards_b, next_states_b, dones_b = map(np.array, zip(*samples))

                # n-step target uses gamma^n; terminal shortened prefixes do not bootstrap.
                gamma_n = cfg.DISCOUNT_FACTOR ** cfg.N_STEP
                target_fn = build_c51_targets if q_estimator.distributional else build_dqn_targets
                targets = target_fn(
                    q_estimator, target_estimator, next_states_b,
                    rewards_b, dones_b, gamma_n, cfg.DOUBLE_DQN,
                )

                # PER: priority controls sampling; IS weights correct the update bias.
                priorities = None
                if cfg.PRIORITIZED_REPLAY:
                    priorities = replay_priorities(
                        q_estimator, states_b, actions_b, targets, cfg.PER_EPS
                    )

                q_estimator.update(
                    states_b, actions_b, targets,
                    weights=weights if cfg.PRIORITIZED_REPLAY else None,
                )
                if cfg.PRIORITIZED_REPLAY:
                    replay.update_priorities(indices, priorities)

            total_t += 1

            # TODO memory
            if mem_profile and total_t % mem_every == 0:
                before = current_process.memory_info().rss / 1024**2

                # Step 1: Force Python cyclic garbage collection.
                collected = gc.collect()
                after_gc = current_process.memory_info().rss / 1024**2

                # Step 2: Ask Linux's glibc allocator to return free pages.
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
                after_trim = current_process.memory_info().rss / 1024**2

                print(
                    f"[GC-TEST] env={env_id} seed={seed} steps={total_t} "
                    f"before={before:.1f}MiB "
                    f"after_gc={after_gc:.1f}MiB "
                    f"after_trim={after_trim:.1f}MiB "
                    f"collected={collected}",
                    flush=True,
                )

                report_memory("train")

            # Validation.
            if total_t >= next_eval_t:
                result = evaluate_policy(
                    eval_env, q_estimator, state_processor,
                    eval_len=cfg.EVAL_LEN, seed=seed + 1,
                    noop_max=reset_noop_max,
                )
                eval_rewards, eval_lengths = result["rewards"], result["lengths"]
                mean_return = float(np.mean(eval_rewards)) if len(eval_rewards) else np.nan

                evaluation_frames.append(total_t)
                evaluation_returns.append(eval_rewards.copy())
                evaluation_mean_returns.append(mean_return)
                np.savez(
                    os.path.join(seed_dir, "evaluation.npz"),
                    frames=np.asarray(evaluation_frames, dtype=np.int64),
                    mean_returns=np.asarray(evaluation_mean_returns, dtype=np.float32),
                    raw_returns=np.asarray(evaluation_returns, dtype=object),
                    seed=seed,
                    game=env_id,
                )

                writer = q_estimator.summary_writer
                if writer is not None and len(eval_rewards):
                    writer.add_scalar("eval/return_mean", mean_return, total_t)
                    writer.add_scalar("eval/return_median", float(np.median(eval_rewards)), total_t)
                    writer.add_scalar("eval/episode_length_mean", float(np.mean(eval_lengths)), total_t)
                    writer.add_scalar("eval/num_episodes", len(eval_rewards), total_t)
                    writer.flush()

                next_eval_t = ((total_t // cfg.EVAL_EVERY) + 1) * cfg.EVAL_EVERY

            if done or total_t >= run_until:
                break
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        i_episode += 1

        # TensorBoard episode/system statistics.
        writer = q_estimator.summary_writer
        if writer is not None:
            writer.add_scalar("episode/epsilon", float(epsilon), i_episode)
            writer.add_scalar("episode/reward", float(episode_reward), i_episode)
            writer.add_scalar("episode/length", int(episode_length), i_episode)
            writer.add_scalar("system/cpu_usage_percent", current_process.cpu_percent(), total_t)
            mem = current_process.memory_info()
            writer.add_scalar("system/memory_rss_mb", mem.rss / 1024**2, total_t)
            writer.add_scalar("system/memory_vms_mb", mem.vms / 1024**2, total_t)
            if cfg.PRIORITIZED_REPLAY:
                writer.add_scalar("replay/beta", beta, total_t)
            writer.flush()

        # Save checkpoint.
        if total_t >= next_checkpoint_t or total_t == cfg.NUM_FRAMES:
            torch.save(
                {
                    "q_estimator": q_estimator.model.state_dict(),
                    "target_estimator": target_estimator.model.state_dict(),
                    "optimizer": q_estimator.optimizer.state_dict(),
                    "global_step": q_estimator.global_step,
                    "total_t": total_t,
                    "episode": i_episode,
                    "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
                    "episode_rewards": np.asarray(episode_rewards, dtype=np.float32),
                    "config": {
                        "env_type": cfg.ENV_TYPE,
                        "network_type": cfg.NETWORK_TYPE,
                        "double_dqn": cfg.DOUBLE_DQN,
                        "n_step": cfg.N_STEP,
                        "prioritized_replay": cfg.PRIORITIZED_REPLAY,
                        "noisy": cfg.NOISY,
                        "num_atoms": cfg.NUM_ATOMS,
                        "v_min": cfg.V_MIN,
                        "v_max": cfg.V_MAX,
                        "adam_lr": cfg.ADAM_LR,
                        "adam_eps": cfg.ADAM_EPS,
                    },
                },
                checkpoint_path,
            )
            next_checkpoint_t = ((total_t // checkpoint_every) + 1) * checkpoint_every

    report_memory("finished")
    stats = EpisodeStats(
        np.asarray(episode_lengths, dtype=np.int64),
        np.asarray(episode_rewards, dtype=np.float32),
    )
    np.savez(
        os.path.join(seed_dir, "training.npz"),
        episode_lengths=stats.episode_lengths,
        episode_rewards=stats.episode_rewards,
        seed=seed,
        game=env_id,
    )
    return stats


###################################################
# Parallel worker
###################################################

def run_job(job):
    """Run one (environment, seed, config) experiment."""
    env_id, seed, config_path, mem_profile, mem_every, stop_after, no_updates = job
    cfg = load_config(config_path)
    validate_config(cfg)
    register_environment_type(cfg)
    # TODO memory
    # if mem_profile:
    #     tracemalloc.start()
    if mem_profile and os.environ.get("DISABLE_TRACEMALLOC") != "1":
        tracemalloc.start()

    torch.set_num_threads(1)
    device = torch.device("cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    name = config_name(cfg)
    game_name = env_id.replace("/", "_")
    # Do not overwrite production metrics/checkpoints during a diagnostic run.
    result_name = (name + "_memprofile" + ("_no_updates" if no_updates else "")) if mem_profile else name
    seed_dir = os.path.abspath(os.path.join("experiments", result_name, game_name, f"seed_{seed}"))
    os.makedirs(seed_dir, exist_ok=True)

    env = eval_env = q_estimator = target_estimator = None
    start = perf_counter()
    print(
        f"[START] env={env_id}, seed={seed}, config={name}, "
        f"pid={os.getpid()}, device={device}",
        flush=True,
    )

    try:
        # AtariPreprocessing receives NOOP_MAX directly.
        env = make_env(env_id, cfg.ENV_TYPE, seed, noop_max=cfg.NOOP_MAX)
        eval_env = make_env(env_id, cfg.ENV_TYPE, seed + 1, noop_max=cfg.NOOP_MAX)

        state_processor = StateProcessor(cfg.ENV_TYPE)

        # Reset once only to determine the network input shape.
        shape_noops = cfg.NOOP_MAX if cfg.ENV_TYPE == "minatar" else 0
        input_shape = state_processor.process(
            env_reset(env, seed=seed, noop_max=shape_noops)
        ).shape
        num_actions = env.action_space.n

        estimator_kwargs = dict(
            input_shape=input_shape,
            num_actions=num_actions,
            env_type=cfg.ENV_TYPE,
            network_type=cfg.NETWORK_TYPE,
            noisy=cfg.NOISY,
            num_atoms=cfg.NUM_ATOMS,
            v_min=cfg.V_MIN,
            v_max=cfg.V_MAX,
            adam_lr=cfg.ADAM_LR,
            adam_eps=cfg.ADAM_EPS,
            device=device,
        )
        q_estimator = Estimator(
            **estimator_kwargs, scope="q_estimator", summaries_dir=seed_dir
        )
        target_estimator = Estimator(
            **estimator_kwargs, scope="target_q"
        )

        stats = train_model(
            env=env,
            eval_env=eval_env,
            env_id=env_id,
            q_estimator=q_estimator,
            target_estimator=target_estimator,
            state_processor=state_processor,
            cfg=cfg,
            seed_dir=seed_dir,
            seed=seed,
            mem_profile=mem_profile,
            mem_every=mem_every,
            stop_after=stop_after,
            no_updates=no_updates,
        )

        elapsed = perf_counter() - start
        print(
            f"[DONE] env={env_id}, seed={seed}, episodes={len(stats.episode_rewards)}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        return {
            "env_id": env_id,
            "seed": seed,
            "seed_dir": seed_dir,
            "episodes": len(stats.episode_rewards),
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        print(
            f"[CRASH] env={env_id}, seed={seed}: {e}",
            flush=True
        )
        raise

    finally:
        for estimator in (q_estimator, target_estimator):
            writer = getattr(estimator, "summary_writer", None)
            if writer is not None:
                writer.flush()
                writer.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


###################################################
# Main
###################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Python config file")
    parser.add_argument(
        "--max-procs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count())),
    )
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--only-env", default=None, help="Run only this environment ID")
    parser.add_argument("--mem-profile", action="store_true", help="Print RSS, USS, traced Python, and replay memory")
    parser.add_argument("--mem-every", type=int, default=5000)
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after N steps per seed (without changing config schedules)")
    parser.add_argument("--no-updates", action="store_true", help="Diagnostic control: collect experience but skip gradient updates")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    validate_config(cfg)

    if args.max_procs < 1 or args.seeds < 1 or args.mem_every < 1:
        raise ValueError("max-procs and seeds must be at least 1")

    env_ids = [args.only_env] if args.only_env else cfg.ENVIRONMENTS
    if args.only_env and args.only_env not in cfg.ENVIRONMENTS:
        parser.error(f"Unknown environment: {args.only_env}")
    jobs = [
        (env_id, seed, config_path, args.mem_profile, args.mem_every, args.stop_after, args.no_updates)
        for env_id, seed in itertools.product(env_ids, range(args.seeds))
    ]
    n_procs = min(args.max_procs, len(jobs))
    name = config_name(cfg)

    print(
        f"--- {name}: {len(env_ids)} environments x "
        f"{args.seeds} seeds with {n_procs} processes ---",
        flush=True,
    )

    start = perf_counter()

    if n_procs == 1:
            # Run ALL environment/seed pairs, not just the first job.
            results = [run_job(job) for job in jobs]
    else:
        ctx = mp.get_context("spawn")
        # Reuse workers for several runs to avoid repeated Python/PyTorch startup.
        # For memory leaks set maxtasksperchild (=1 e.g.)
        with ctx.Pool(n_procs) as pool:
            results = pool.map(run_job, jobs, chunksize=1)

    print(
        f"--- Finished {len(jobs)} runs in {perf_counter() - start:.1f} seconds ---",
        flush=True,
    )
