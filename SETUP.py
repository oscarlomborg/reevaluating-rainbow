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

def env_reset(env, seed=None, noop_max=0):
    """Reset environment and optionally perform random NOOP actions."""
    obs, _ = env.reset(seed=seed)

    if noop_max > 0:
        num_noops = np.random.randint(0, noop_max + 1)

        for _ in range(num_noops):
            obs, _, terminated, truncated, _ = env.step(0)

            if terminated or truncated:
                obs, _ = env.reset()

    return obs


def env_step(env, action):
    """Gymnasium step -> obs, reward, done, info."""
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, terminated or truncated, info


###################################################
# MinAtar rendering
###################################################

class ScaleRender(gym.Wrapper):
    """Upscale MinAtar RGB rendering with nearest-neighbour."""

    def __init__(self, env, scale=30):
        super().__init__(env)
        self.scale = scale

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None

        frame = np.asarray(frame)

        # MinAtar RGB is usually float in [0,1]
        if np.issubdtype(frame.dtype, np.floating):
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # e.g. 10x10 -> 300x300 for scale=30
        return np.repeat(
            np.repeat(frame, self.scale, axis=0),
            self.scale,
            axis=1
        )


###################################################
# State processing
###################################################

class StateProcessor:
    """
    Convert observations to CHW uint8 format.

    MinAtar:
        HWC (10,10,C) -> CHW (C,10,10)

    Atari:
        assumes Atari preprocessing/frame stacking has already
        produced 84x84 observations.
    """

    def __init__(self, env_type):
        if env_type not in ("minatar", "atari"):
            raise ValueError(f"Unknown env_type: {env_type}")

        self.env_type = env_type


    def process(self, state):
        state = np.asarray(state)

        if state.ndim != 3:
            raise ValueError(
                f"Expected 3-D observation, got {state.shape}"
            )

        if self.env_type == "minatar":
            # Gym MinAtar: HWC -> CHW
            state = np.transpose(state, (2, 0, 1))

        elif self.env_type == "atari":
            # Atari network expects 84x84 input
            if state.shape[-2:] == (84, 84):
                pass                        # already CHW
            elif state.shape[:2] == (84, 84):
                state = np.transpose(state, (2, 0, 1))  # HWC -> CHW
            else:
                raise ValueError(
                    "Atari observation must already be preprocessed "
                    f"to 84x84, got {state.shape}"
                )

        return state.astype(np.uint8, copy=False)


###################################################
# Network architectures
###################################################

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable means
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable noise scales
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma0 = sigma0

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        p = self.in_features

        bound = 1 / np.sqrt(p)

        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        self.weight_sigma.data.fill_(
            self.sigma0 / np.sqrt(p)
        )
        self.bias_sigma.data.fill_(
            self.sigma0 / np.sqrt(p)
        )

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)

        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = (self.weight_mu + self.weight_sigma * self.weight_epsilon)

            bias = (self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

def get_linear(noisy):
    return NoisyLinear if noisy else nn.Linear

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_shape, env_type):
        super().__init__()

        self.env_type = env_type
        channels, height, width = input_shape

        if env_type == "minatar":
            self.conv1 = nn.Conv2d(channels, 16, 3, stride=1)

            conv_h = height - 2
            conv_w = width - 2

            self.feature_dim = 16 * conv_h * conv_w

        elif env_type == "atari":
            self.conv1 = nn.Conv2d(channels, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

            self.feature_dim = 64 * 7 * 7

    def extract_features(self, x):
        x = x.float()

        if self.env_type == "minatar":
            x = F.relu(self.conv1(x))

        elif self.env_type == "atari":
            x = x / 255.0
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

        return torch.flatten(x, start_dim=1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DQN(ConvFeatureExtractor):

    def __init__(self, input_shape, num_actions, env_type, noisy=False):
        super().__init__(input_shape, env_type)

        hidden = 128 if env_type == "minatar" else 512

        Linear = get_linear(noisy)

        self.fc1 = Linear(self.feature_dim, hidden)

        self.fc2 = Linear(hidden, num_actions)

    def forward(self, x):
        x = self.extract_features(x)

        x = F.relu(self.fc1(x))

        return self.fc2(x)

class DuelDQN(ConvFeatureExtractor):

    def __init__(self, input_shape, num_actions, env_type, noisy=False):
        super().__init__(input_shape, env_type)

        hidden = 128 if env_type == "minatar" else 512

        Linear = get_linear(noisy)

        self.value_fc = Linear(self.feature_dim, hidden)
        self.value_out = Linear(hidden, 1)

        self.adv_fc = Linear(self.feature_dim, hidden)
        self.adv_out = Linear(hidden, num_actions)

    def forward(self, x):
        x = self.extract_features(x)

        value = self.value_out(F.relu(self.value_fc(x)))
        advantage = self.adv_out(F.relu(self.adv_fc(x)))

        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DistributionalDQN(ConvFeatureExtractor):

    def __init__(self, input_shape, num_actions, env_type, num_atoms=51, noisy=False):
        super().__init__(input_shape, env_type)

        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.noisy = noisy

        if env_type == "minatar":
            hidden = 128 
            self.v_min, self.v_max = (-10.0, 10.0)

        elif env_type == "atari":
            hidden = 512
            self.v_min, self.v_max = (-10.0, 10.0)

        Linear = get_linear(noisy)

        # Fixed atom support z_i
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, num_atoms))

        self.fc1 = Linear(self.feature_dim, hidden)

        self.fc2 = Linear(hidden, self.num_actions * self.num_atoms)

    def dist(self, x):

        x = self.extract_features(x)

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        logits = logits.view(-1, self.num_actions, self.num_atoms)

        probs = F.softmax(logits, dim=2)

        return probs


    def forward(self, x):

        probs = self.dist(x)
        q_values = torch.sum(probs * self.support.view(1, 1, -1),dim=2)

        return q_values

class DistributionalDuelDQN(ConvFeatureExtractor):

    def __init__(self, input_shape, num_actions, env_type, num_atoms=51, noisy=False):
        super().__init__(input_shape, env_type)

        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.noisy = noisy

        if env_type == "minatar":
            hidden = 128 
            self.v_min, self.v_max = (0, 100.0)

        elif env_type == "atari":
            hidden = 512
            self.v_min, self.v_max = (-10.0, 10.0)

        Linear = get_linear(noisy)

        # Fixed atom support z_i
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, num_atoms))

        self.value_fc = Linear(self.feature_dim, hidden)
        self.value_out = Linear(hidden,num_atoms)

        self.adv_fc = Linear(self.feature_dim, hidden)
        self.adv_out = Linear(hidden, num_actions * num_atoms)

    def dist(self, x):

        x = self.extract_features(x)
        
        value = self.value_out(F.relu(self.value_fc(x)))
        value = value.view(-1, 1, self.num_atoms)

        advantage = self.adv_out(F.relu(self.adv_fc(x)))
        advantage = advantage.view(-1, self.num_actions, self.num_atoms)

        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
        probs = F.softmax(logits, dim=2)

        return probs

    def forward(self, x):

        probs = self.dist(x)
        q_values = torch.sum(probs * self.support.view(1, 1, -1), dim=2)

        return q_values

###################################################
# Network factory
###################################################

def build_network(input_shape, num_actions, env_type, network_type="dqn",
                  noisy=False, num_atoms=51):

    nets = {
        "dqn": DQN,
        "dueling": DuelDQN,
        "distributional": DistributionalDQN,
        "distributional_dueling": DistributionalDuelDQN,
    }

    if network_type not in nets:
        raise ValueError(f"Unknown network_type: {network_type}")

    kwargs = dict(input_shape=input_shape, num_actions=num_actions,
                  env_type=env_type, noisy=noisy)

    if "distributional" in network_type:
        kwargs["num_atoms"] = num_atoms

    return nets[network_type](**kwargs)


###################################################
# Estimator
###################################################

class Estimator:

    def __init__(self, input_shape, num_actions, env_type,
                 network_type="dqn", noisy=False, num_atoms=51,
                 scope="estimator", summaries_dir=None, device=None,
                 learning_rate=0.00025):

        self.scope = scope
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noisy = noisy
        self.distributional = "distributional" in network_type

        self.model = build_network(
            input_shape, num_actions, env_type,
            network_type, noisy, num_atoms
        ).to(self.device)

        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=learning_rate,
            alpha=0.99, eps=1e-6, momentum=0.95
        )

        self.global_step = 0
        self.summary_writer = None

        if summaries_dir:
            path = os.path.join(summaries_dir, f"summaries_{scope}")
            os.makedirs(path, exist_ok=True)
            self.summary_writer = SummaryWriter(path)


    def _to_tensor(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.uint8, device=self.device)


    def _set_mode(self, use_noise):
        # Noise is only active while model.training == True
        if use_noise and self.noisy:
            self.model.train()
            self.model.reset_noise()
        else:
            self.model.eval()


    @torch.no_grad()
    def predict(self, states, use_noise=False):
        # Return expected Q-values
        old_mode = self.model.training
        self._set_mode(use_noise)

        q = self.model(self._to_tensor(states))

        self.model.train(old_mode)
        return q.cpu().numpy()


    @torch.no_grad()
    def predict_dist(self, states, use_noise=False):
        # Return p(z_i | s,a)
        if not self.distributional:
            raise RuntimeError("Network is not distributional.")

        old_mode = self.model.training
        self._set_mode(use_noise)

        probs = self.model.dist(self._to_tensor(states))

        self.model.train(old_mode)
        return probs.cpu().numpy()


    def update(self, states, actions, targets, weights=None):

        self.model.train()

        # Fresh noise for this gradient update
        if self.noisy:
            self.model.reset_noise()

        states = self._to_tensor(states)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        if self.distributional:
            # Cross entropy between projected target and p(z|s,a)
            probs = self.model.dist(states)
            idx = torch.arange(len(actions), device=self.device)
            action_probs = probs[idx, actions]

            losses = -(targets * torch.log(action_probs.clamp(min=1e-8))).sum(dim=1)

            # Only needed for logging
            q_values = (probs * self.model.support.view(1, 1, -1)).sum(dim=2)

        else:
            # Standard squared TD error
            q_values = self.model(states)
            action_q = q_values.gather(1, actions[:, None]).squeeze(1)
            losses = (targets - action_q).pow(2)

        # Prioritized replay importance weights
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            loss = (weights * losses).mean()
        else:
            loss = losses.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        # TensorBoard
        if self.summary_writer and self.global_step % 100 == 0:
            s = self.global_step
            self.summary_writer.add_scalar("train/loss", loss.item(), s)
            self.summary_writer.add_scalar("q_values/mean", q_values.mean().item(), s)
            self.summary_writer.add_scalar("q_values/max", q_values.max().item(), s)

        return loss.item()


###################################################
# Standard / Double DQN targets
###################################################

@torch.no_grad()
def build_dqn_targets(online, target, next_states, rewards, dones,
                      gamma, double_dqn=True):

    x = online._to_tensor(next_states)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=online.device)
    d = torch.as_tensor(dones, dtype=torch.float32, device=online.device)

    # Fresh noise for target calculation
    online._set_mode(online.noisy)
    target._set_mode(target.noisy)

    if double_dqn:
        # Online chooses, target evaluates
        actions = online.model(x).argmax(dim=1)
        next_q = target.model(x).gather(1, actions[:, None]).squeeze(1)
    else:
        next_q = target.model(x).max(dim=1).values

    return (r + gamma * (1 - d) * next_q).cpu().numpy()


###################################################
# C51 projection
###################################################

def categorical_projection(next_dist, rewards, dones, gamma, support):

    device = next_dist.device
    r = torch.as_tensor(rewards, dtype=torch.float32, device=device)[:, None]
    d = torch.as_tensor(dones, dtype=torch.float32, device=device)[:, None]

    v_min, v_max = support[0], support[-1]
    dz = (v_max - v_min) / (len(support) - 1)

    # Bellman transformed atoms
    tz = (r + gamma * (1 - d) * support[None, :]).clamp(v_min, v_max)

    # Position between support atoms
    b = (tz - v_min) / dz
    l, u = b.floor().long(), b.ceil().long()

    # Probability weights for lower/upper neighbours
    wl, wu = u.float() - b, b - l.float()

    # Exact atom -> all mass remains there
    same = l == u
    wl = torch.where(same, torch.ones_like(wl), wl)
    wu = torch.where(same, torch.zeros_like(wu), wu)

    projected = torch.zeros_like(next_dist)
    projected.scatter_add_(1, l, next_dist * wl)
    projected.scatter_add_(1, u, next_dist * wu)

    return projected


###################################################
# Distributional / C51 targets
###################################################

@torch.no_grad()
def build_c51_targets(online, target, next_states, rewards, dones,
                      gamma, double_dqn=True):

    if not online.distributional or not target.distributional:
        raise ValueError("C51 requires distributional networks.")

    x = online._to_tensor(next_states)

    # Fresh NoisyNet samples
    online._set_mode(online.noisy)
    target._set_mode(target.noisy)

    # Target-network distribution p(z | s',a)
    target_probs = target.model.dist(x)

    if double_dqn:
        # Online chooses next action
        actions = online.model(x).argmax(dim=1)
    else:
        # Target chooses from expected Q-values
        actions = target.model(x).argmax(dim=1)

    # Distribution corresponding to selected action
    idx = torch.arange(len(x), device=online.device)
    next_dist = target_probs[idx, actions]

    projected = categorical_projection(
        next_dist, rewards, dones, gamma,
        target.model.support
    )

    return projected.cpu().numpy()


###################################################
# Target network copy
###################################################

class ModelParametersCopier:

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def make(self):
        self.target.model.load_state_dict(self.source.model.state_dict())


###################################################
# Epsilon-greedy policy
###################################################

def make_epsilon_greedy_policy(estimator, num_actions,
                               use_noisy_exploration=False):

    def policy_fn(observation, epsilon):

        probs = np.full(num_actions, epsilon / num_actions)

        q = estimator.predict(
            np.expand_dims(observation, 0),
            use_noise=use_noisy_exploration
        )[0]

        probs[np.argmax(q)] += 1 - epsilon
        return probs / probs.sum()

    return policy_fn


###################################################
# Policy evaluation
###################################################

def evaluate_policy(env, q_estimator, state_processor,
                    eval_len, epsilon=0.0, seed=None, noop_max=0):

    state = state_processor.process(env_reset(env, seed=seed, noop_max=noop_max))

    episode_reward = episode_length = 0
    rewards, lengths = [], []

    for _ in range(eval_len):

        # Evaluation -> no NoisyNet noise
        q = q_estimator.predict(np.expand_dims(state, 0), use_noise=False)[0]

        action = (
            np.random.randint(env.action_space.n)
            if np.random.rand() < epsilon
            else int(np.argmax(q))
        )

        next_state, reward, done, _ = env_step(env, action)
        next_state = state_processor.process(next_state)

        episode_reward += reward
        episode_length += 1

        if done:
            rewards.append(episode_reward)
            lengths.append(episode_length)

            state = state_processor.process(env_reset(env, seed=seed, noop_max=noop_max))
            episode_reward = episode_length = 0
        else:
            state = next_state

    return {
        "rewards": np.asarray(rewards, dtype=np.float32),
        "lengths": np.asarray(lengths, dtype=np.int64),
    }