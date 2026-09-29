"""
Deep Q-Network (DQN) implementation for the Gymnasium CartPole-v1 environment.

Implements the algorithm discussed in class:
  - A small MLP Q-network q(s, ; phi) -> R^{|A|}
  - A replay buffer B
  - A (soft-updated) target network phi'
  - epsilon-greedy exploration
  - The semi-gradient update:
        y_i = r_i + gamma * max_a' q(s_i', a'; phi')      (0 if s_i' terminal)
        phi <- phi - alpha * grad_phi [ 1/(2B) sum_i (y_i - q(s_i,a_i;phi))^2 ]
        phi' <- phi' + tau * (phi - phi')

Usage:
    python dqn.py
"""
from __future__ import annotations

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from plot_data import plot_data

import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Deque, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-capacity FIFO buffer B storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = 2000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        """Sample a random minibatch and collate it into batched tensors."""
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """q_hat(s, .; phi): R^d -> R^{|A|}, a small MLP outputting all action-values at once."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    hidden_dim: int = 30
    buffer_capacity: int = 2000
    batch_size: int = 32
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.08        # soft target-network update rate
    eps_max: float = 1.0
    eps_min: float = 0.01
    eps_delay: float = 300
    eps_miniter: float = 5000
    min_buffer_size: int = 500   # steps to collect before training starts
    grad_updates_per_step: int = 1  # "K" from the lecture slides
    max_grad_norm: float = 10.0
    seed: int | None = None


class DQNAgent:
    """
    Deep Q-learning agent with experience replay and a soft-updated target network.

    Mirrors the algorithm:
        1) Initialize B and make a copy phi' <- phi of the weights
        2) At step t observe (s_t, a_t, r_{t+1}, s_{t+1}) and add it to B
        3) Repeat K times:
             a) Sample a batch from B
             b) Set y_i = r_i + gamma * max_a' q_phi'(s_i', a')   (0 if terminal)
             c) phi <- phi - alpha * grad_phi [ 1/(2B) sum (y_i - q_phi(s_i,a_i))^2 ]
        4) Update phi' <- phi' + tau * (phi - phi')
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = DQNConfig()):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = config
        self.total_steps = 0

        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network (phi) and target network (phi'), initialized as a copy of phi.
        self.q_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_capacity)

        self.epsilon = config.eps_max

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """epsilon-greedy action selection over q_hat(s, .; phi)."""
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)  # one forward pass -> all |A| action-values
            return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        self.epsilon = min(max([self.cfg.eps_max- ((self.total_steps - self.cfg.eps_delay)/self.cfg.eps_miniter)*(self.cfg.eps_max - self.cfg.eps_min), self.cfg.eps_min]), self.cfg.eps_max)

    # ------------------------------------------------------------------ #
    # Storing experience
    # ------------------------------------------------------------------ #
    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def update(self) -> float | None:
        """
        Perform K gradient steps (config.grad_updates_per_step) using minibatches
        sampled from the replay buffer, then soft-update the target network.

        Returns the last minibatch's loss value (for logging), or None if the
        buffer does not yet contain enough transitions to train on.
        """
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.min_buffer_size):
            return None

        last_loss = None
        for _ in range(self.cfg.grad_updates_per_step):
            last_loss = self._gradient_step()

        self._soft_update_target()
        return last_loss

    def _gradient_step(self) -> float:
        batch = self.buffer.sample(self.cfg.batch_size)

        states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # q_phi(s_i, a_i): gather the Q-value of the action actually taken.
        q_values = self.q_net(states).gather(1, actions)

        # y_i = r_i + gamma * max_a' q_phi'(s_i', a')   (bootstrap using the TARGET network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.cfg.gamma * next_q_values * (1.0 - dones)

        # Semi-gradient loss: 1/(2B) sum (y_i - q_phi(s_i,a_i))^2  == 0.5 * MSE
        loss = 0.5 * F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def _soft_update_target(self) -> None:
        """phi' <- phi' + tau * (phi - phi')"""
        tau = self.cfg.tau
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


# ---------------------------------------------------------------------- #
# Training loop for CartPole-v1
# ---------------------------------------------------------------------- #
def train_cartpole(
    num_episodes: int = 300,
    render: bool = False,
    config: DQNConfig = DQNConfig(),
) -> Tuple[DQNAgent, List[float]]:
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config)
    episode_rewards: List[float] = []
 
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=config.seed)
        episode_reward = 0.0
        done = False
 
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
 
            # Only bootstrap-terminal on true termination, not on time-limit truncation.
            agent.store_transition(state, action, reward, next_state, terminated)
 
            loss = agent.update()

            state = next_state
            episode_reward += reward
 
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # if episode % 20 == 0:
        #     print(
        #         f"Episode {episode:4d} | "
        #         f"epsilon {agent.epsilon:.3f}"
        #     )
         
 
    env.close()
    return agent, episode_rewards

if __name__ == "__main__":

    n_runs = 28
    num_episodes = 200
    all_rewards = np.zeros((n_runs, num_episodes))

    for run in range(n_runs):
        print(f'Run {run+1}/{n_runs}')
        config = DQNConfig(seed=run)
        _, rewards = train_cartpole(num_episodes=num_episodes, config=config)
        all_rewards[run] = rewards

    plot_data(all_rewards, save_path="cartpole_training_curve_tau008.png")
