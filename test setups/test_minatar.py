"""Smoke test for a MinAtar installation using its Gymnasium interface."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

import gymnasium as gym
import minatar  # noqa: F401 - importing registers the Gymnasium environments
import numpy as np
from minatar.gym import register_envs


# Some Gymnasium versions do not automatically load MinAtar's plugin entry
# point, so register the environments explicitly.
register_envs()


ENV_IDS = (
    "MinAtar/Asterix-v1",
    "MinAtar/Breakout-v1",
    "MinAtar/Freeway-v1",
    "MinAtar/Seaquest-v1",
    "MinAtar/SpaceInvaders-v1",
)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def test_environment(env_id: str, steps: int = 100) -> None:
    env = gym.make(env_id)
    try:
        observation, info = env.reset(seed=42)

        assert isinstance(observation, np.ndarray)
        assert observation.shape[:2] == (10, 10), observation.shape
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)

        episodes = 0
        total_reward = 0.0

        for _ in range(steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(observation)
            assert isinstance(info, dict)
            total_reward += float(reward)

            if terminated or truncated:
                episodes += 1
                observation, info = env.reset()

        print(
            f"PASS  {env_id:<30} "
            f"obs={observation.shape!s:<12} "
            f"actions={env.action_space.n} "
            f"reward={total_reward:g} episodes={episodes}"
        )
    finally:
        env.close()


def main() -> int:
    print(f"Python:    {sys.version.split()[0]}")
    print(f"Gymnasium: {package_version('gymnasium')}")
    print(f"MinAtar:   {package_version('minatar')}\n")

    failures: list[tuple[str, Exception]] = []
    for env_id in ENV_IDS:
        try:
            test_environment(env_id)
        except Exception as exc:  # continue so every environment is checked
            failures.append((env_id, exc))
            print(f"FAIL  {env_id}: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)} of {len(ENV_IDS)} environments failed.")
        return 1

    print(f"\nAll {len(ENV_IDS)} MinAtar environments work correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
