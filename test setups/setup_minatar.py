import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


###################################################
# Environment helpers
###################################################

def env_reset(env, seed=None):
    """Return only the observation from Gymnasium reset()."""
    obs, _ = env.reset(seed=seed)
    return obs


def env_step(env, action):
    """Return (obs, reward, done, info) from the Gymnasium step API."""
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, bool(terminated or truncated), info


class ScaleRender(gym.Wrapper):
    """Convert MinAtar render to uint8 and upscale with nearest-neighbour."""

    def __init__(self, env, scale=30):
        super().__init__(env)
        self.scale = scale

    def render(self):
        frame = self.env.render()

        if frame is None:
            return None

        frame = np.asarray(frame)

        # MinAtar rgb_array is float RGB in [0, 1].
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # 10x10 -> 200x200 when scale=20
        frame = np.repeat(frame, self.scale, axis=0)
        frame = np.repeat(frame, self.scale, axis=1)

        return frame

###################################################
# MinAtar state processing
###################################################

class StateProcessor:
    """Convert a MinAtar observation from HWC to CHW uint8 format.

    MinAtar observations are small binary feature maps, typically shaped
    (10, 10, C). No Atari cropping, grayscale conversion, resizing, or
    four-frame stacking is needed.
    """

    def process(self, state):
        state = np.asarray(state)
        if state.ndim != 3:
            raise ValueError(f"Expected a 3-D MinAtar observation, got shape {state.shape}")

        # Gym MinAtar exposes observations as (H, W, C); PyTorch Conv2d expects (C, H, W).
        state = np.transpose(state, (2, 0, 1))
        return state.astype(np.uint8, copy=False)


###################################################
# Agent
###################################################

class QNetwork(nn.Module):
    """Small convolutional Q-network for MinAtar observations."""

    def __init__(self, input_shape, num_actions):
        super().__init__()
        channels, height, width = input_shape

        # Common compact MinAtar-style architecture: one 3x3 convolution followed by an MLP.
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1)

        conv_h = height - 3 + 1
        conv_w = width - 3 + 1
        if conv_h <= 0 or conv_w <= 0:
            raise ValueError(f"Observation shape {input_shape} is too small for a 3x3 convolution")

        self.fc1 = nn.Linear(16 * conv_h * conv_w, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        # MinAtar feature planes are binary (0/1), so do not divide by 255.
        x = x.float()
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Estimator:
    """Wrap a MinAtar QNetwork with optimizer and optional TensorBoard logging."""

    def __init__(self, input_shape, num_actions, scope="estimator", summaries_dir=None,
                 device=None, learning_rate=0.00025):
        self.scope = scope
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(input_shape=input_shape, num_actions=num_actions).to(self.device)

        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=learning_rate,
            alpha=0.99,
            eps=1e-6,
            momentum=0.95,
        )
        self.global_step = 0

        self.summary_writer = None
        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, f"summaries_{scope}")
            os.makedirs(summary_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(summary_dir)

    def _to_tensor(self, states):
        return torch.as_tensor(np.asarray(states), dtype=torch.uint8, device=self.device)

    @torch.no_grad()
    def predict(self, states):
        self.model.eval()
        return self.model(self._to_tensor(states)).cpu().numpy()

    def update(self, states, actions, targets):
        self.model.train()

        states = self._to_tensor(states)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=self.device)

        predictions = self.model(states)
        action_predictions = predictions.gather(1, actions.unsqueeze(1)).squeeze(1)

        # MSE, matching your previous implementation.
        losses = (targets - action_predictions).pow(2)
        loss = losses.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        if self.summary_writer:
            step = self.global_step

            # Scalars are cheap enough to log fairly often.
            if step % 100 == 0:
                self.summary_writer.add_scalar("train/loss", loss.item(), step)
                self.summary_writer.add_scalar("q_values/mean", predictions.mean().item(), step)
                self.summary_writer.add_scalar("q_values/max", predictions.max().item(), step)

        return loss.item()


class ModelParametersCopier:
    """Copy model parameters from one estimator to another."""

    def __init__(self, estimator1, estimator2):
        self.estimator1 = estimator1
        self.estimator2 = estimator2

    def make(self):
        self.estimator2.model.load_state_dict(self.estimator1.model.state_dict())


def make_epsilon_greedy_policy(estimator, num_actions):
    """Return an epsilon-greedy policy over direct MinAtar action indices."""

    def policy_fn(observation, epsilon):
        probs = np.full(num_actions, epsilon / num_actions, dtype=np.float64)
        q_values = estimator.predict(np.expand_dims(observation, 0))[0]
        best_action = int(np.argmax(q_values))
        probs[best_action] += 1.0 - epsilon
        # Protect against floating-point rounding errors.
        probs /= probs.sum()
        return probs

    return policy_fn


###################################################
# Policy evaluation
###################################################

def evaluate_policy(
    env,
    q_estimator,
    state_processor,
    eval_len,
    epsilon=0.0,
    seed=None,
):
    state = state_processor.process(
        env_reset(env, seed=seed)
    )

    episode_reward = 0.0
    episode_length = 0

    episode_rewards = []
    episode_lengths = []

    num_actions = env.action_space.n

    for _ in range(eval_len):

        q_values = q_estimator.predict(
            np.expand_dims(state, axis=0)
        )[0]

        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = int(np.argmax(q_values))

        next_state, reward, done, _ = env_step(
            env,
            action,
        )

        next_state = state_processor.process(next_state)

        episode_reward += reward
        episode_length += 1

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            state = state_processor.process(
                env_reset(env)
            )

            episode_reward = 0.0
            episode_length = 0

        else:
            state = next_state

    return {
        "rewards": np.asarray(
            episode_rewards,
            dtype=np.float32,
        ),
        "lengths": np.asarray(
            episode_lengths,
            dtype=np.int64,
        ),
    }