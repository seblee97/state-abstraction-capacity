"""Deep Successor Reinforcement Learning (DSR).

Implements successor representations learned via deep networks, following the
architecture patterns of DQN and PPO in this codebase.

Reference:
    Kulkarni et al., "Deep Successor Reinforcement Learning" (2016)
"""

from sac.models import base

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ConvDSRNet(nn.Module):
    """Convolutional network for Deep Successor Representations.

    Outputs successor features for each action. Optionally includes a
    reconstruction head for auxiliary loss.
    """

    def __init__(self, num_actions, feature_dim=128, reconstruction=False,
                 state_shape=None, optimistic_init=0.0):
        super(ConvDSRNet, self).__init__()
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.reconstruction = reconstruction
        self._state_shape = state_shape

        # Encoder: state -> features (phi)
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # Assuming single-channel input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, feature_dim)

        # Successor features: one SR vector per action
        # Output shape: (batch, num_actions, feature_dim)
        self.sr_fc = nn.Linear(feature_dim, num_actions * feature_dim)

        # Reward weights: w such that r = phi(s) @ w
        self.reward_weights = nn.Parameter(torch.zeros(feature_dim))
        nn.init.uniform_(self.reward_weights, -0.01, 0.01)

        # Optimistic initialization for Q-values
        if optimistic_init > 0:
            # Initialize reward weights to produce optimistic values
            nn.init.constant_(self.reward_weights, optimistic_init / feature_dim)

        # Optional reconstruction head: features -> state
        # Decoder will be lazily built on first reconstruct() call once we know conv shape
        self._decoder_built = False

    def _get_features(self, x):
        """Encode state to feature representation phi(s)."""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        self._conv_shape = x.shape[1:]  # Save for decoder
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on features
        return x

    def forward(self, x):
        """Forward pass returning successor features and state features.

        Returns:
            sr: Successor features of shape (batch, num_actions, feature_dim)
            phi: State features of shape (batch, feature_dim)
        """
        phi = self._get_features(x)
        sr = self.sr_fc(phi)
        sr = sr.view(-1, self.num_actions, self.feature_dim)
        return sr, phi

    def get_q_values(self, x):
        """Compute Q-values from successor features: Q(s,a) = psi(s,a) @ w."""
        sr, phi = self.forward(x)
        # sr: (batch, num_actions, feature_dim)
        # reward_weights: (feature_dim,)
        q = torch.einsum('baf,f->ba', sr, self.reward_weights)
        return q

    def _build_decoder(self, device):
        """Build decoder layers once we know the conv shape."""
        c, h, w = self._conv_shape
        self.decoder_fc1 = nn.Linear(self.feature_dim, 128).to(device)
        self.decoder_fc2 = nn.Linear(128, c * h * w).to(device)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1).to(device)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1).to(device)
        self._decoder_built = True

    def reconstruct(self, phi):
        """Reconstruct state from features (if reconstruction head enabled)."""
        if not self.reconstruction:
            raise RuntimeError("Reconstruction head not enabled")

        # Lazily build decoder on first call (after we know _conv_shape from forward pass)
        if not self._decoder_built:
            self._build_decoder(phi.device)

        x = torch.relu(self.decoder_fc1(phi))
        x = self.decoder_fc2(x)

        # Reshape to match conv output shape
        batch_size = phi.size(0)
        x = x.view(batch_size, *self._conv_shape)

        x = torch.relu(self.deconv1(x))
        x = self.deconv2(x)  # No activation on reconstruction
        return x


class FFDSRNet(nn.Module):
    """Feedforward network for Deep Successor Representations.

    Outputs successor features for each action. Optionally includes a
    reconstruction head for auxiliary loss.
    """

    def __init__(self, input_dim, num_actions, feature_dim=128,
                 reconstruction=False, optimistic_init=0.0):
        super(FFDSRNet, self).__init__()
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.reconstruction = reconstruction
        self.input_dim = input_dim

        # Encoder: state -> features (phi)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, feature_dim)

        # Successor features: one SR vector per action
        # Output shape: (batch, num_actions, feature_dim)
        self.sr_fc = nn.Linear(feature_dim, num_actions * feature_dim)

        # Reward weights: w such that r = phi(s) @ w
        self.reward_weights = nn.Parameter(torch.zeros(feature_dim))
        nn.init.uniform_(self.reward_weights, -0.01, 0.01)

        # Optimistic initialization for Q-values
        if optimistic_init > 0:
            nn.init.constant_(self.reward_weights, optimistic_init / feature_dim)

        # Optional reconstruction head: features -> state
        if reconstruction:
            self.decoder_fc1 = nn.Linear(feature_dim, 128)
            self.decoder_fc2 = nn.Linear(128, 128)
            self.decoder_fc3 = nn.Linear(128, input_dim)

    def _get_features(self, x):
        """Encode state to feature representation phi(s)."""
        if isinstance(x, tuple):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on features
        return x

    def forward(self, x):
        """Forward pass returning successor features and state features.

        Returns:
            sr: Successor features of shape (batch, num_actions, feature_dim)
            phi: State features of shape (batch, feature_dim)
        """
        phi = self._get_features(x)
        sr = self.sr_fc(phi)
        sr = sr.view(-1, self.num_actions, self.feature_dim)
        return sr, phi

    def get_q_values(self, x):
        """Compute Q-values from successor features: Q(s,a) = psi(s,a) @ w."""
        sr, phi = self.forward(x)
        # sr: (batch, num_actions, feature_dim)
        # reward_weights: (feature_dim,)
        q = torch.einsum('baf,f->ba', sr, self.reward_weights)
        return q

    def reconstruct(self, phi):
        """Reconstruct state from features (if reconstruction head enabled)."""
        if not self.reconstruction:
            raise RuntimeError("Reconstruction head not enabled")
        x = torch.relu(self.decoder_fc1(phi))
        x = torch.relu(self.decoder_fc2(x))
        x = self.decoder_fc3(x)  # No activation on reconstruction
        return x


class ReplayBuffer:
    """Simple replay buffer for experience replay."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, active):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, active)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, actives = zip(
            *[self.buffer[idx] for idx in batch]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(actives),
        )

    def __len__(self):
        return len(self.buffer)


class DSR(base.BaseModel):
    """Deep Successor Reinforcement Learning.

    Learns successor representations using a deep network, with optional
    reconstruction auxiliary loss.

    Args:
        sample_state: Example state for inferring input dimensions
        num_actions: Number of discrete actions
        batch_size: Minibatch size for training
        learning_rate: Learning rate for optimizer
        discount_factor: Gamma for successor representation TD learning
        exploration_rate: Initial epsilon for epsilon-greedy exploration
        exploration_decay: Multiplicative decay for exploration rate
        target_update_frequency: Steps between target network updates
        replay_buffer_size: Capacity of replay buffer
        burnin: Minimum buffer size before training starts
        feature_dim: Dimension of successor feature vectors
        reconstruction: Whether to include reconstruction auxiliary loss
        reconstruction_coef: Weight for reconstruction loss
        convolutional: Whether to use convolutional architecture
        optimistic_init: Optimistic initialization value for Q-values
        weight_decay: L2 regularization coefficient
    """

    def __init__(
            self,
            sample_state,
            num_actions,
            batch_size: int,
            learning_rate: float,
            discount_factor: float,
            exploration_rate: float,
            exploration_decay: float,
            target_update_frequency: int,
            replay_buffer_size: int,
            burnin: int,
            feature_dim: int = 128,
            reconstruction: bool = False,
            reconstruction_coef: float = 0.1,
            convolutional: bool = False,
            optimistic_init: float = 0.0,
            weight_decay: float = 0.0,
        ):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(sample_state, tuple):
            sample_state = np.array(sample_state)

        if convolutional:
            state_shape = sample_state.shape[1:]  # cut batch dimension
            net = ConvDSRNet(
                num_actions=num_actions,
                feature_dim=feature_dim,
                reconstruction=reconstruction,
                state_shape=state_shape,
                optimistic_init=optimistic_init
            )
            self._target_net = ConvDSRNet(
                num_actions=num_actions,
                feature_dim=feature_dim,
                reconstruction=False,  # Target net doesn't need reconstruction
                state_shape=state_shape
            ).to(self._device)
            self._state_shape = state_shape
        else:
            input_dim = len(sample_state.flatten())
            net = FFDSRNet(
                input_dim=input_dim,
                num_actions=num_actions,
                feature_dim=feature_dim,
                reconstruction=reconstruction,
                optimistic_init=optimistic_init
            )
            self._target_net = FFDSRNet(
                input_dim=input_dim,
                num_actions=num_actions,
                feature_dim=feature_dim,
                reconstruction=False  # Target net doesn't need reconstruction
            ).to(self._device)
            self._state_shape = (input_dim,)

        self._net = net.to(self._device)

        # Load state dict, filtering out reconstruction layers if target doesn't have them
        self._sync_target_network()
        self._target_net.eval()

        self._num_actions = num_actions
        self._feature_dim = feature_dim
        self._reconstruction = reconstruction
        self._reconstruction_coef = reconstruction_coef

        self._optimizer = optim.AdamW(self._net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self._buffer = ReplayBuffer(capacity=replay_buffer_size)

        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay
        self._target_update_frequency = target_update_frequency
        self._burnin = burnin
        self._step_count = 0

        super().__init__()

    def _sync_target_network(self):
        """Sync target network weights, filtering out reconstruction layers."""
        source_dict = self._net.state_dict()
        target_dict = self._target_net.state_dict()
        # Only copy keys that exist in target
        filtered_dict = {k: v for k, v in source_dict.items() if k in target_dict}
        self._target_net.load_state_dict(filtered_dict)

    def select_action(self, state):
        """Select action using epsilon-greedy exploration."""
        if isinstance(state, tuple):
            state = np.array(state)
        if np.random.rand() < self._exploration_rate:
            action = np.random.choice(range(self._num_actions))
        else:
            action = self.select_greedy_action(state)
        return action

    def select_greedy_action(self, state):
        """Select action greedily based on Q-values."""
        if isinstance(state, tuple):
            state = np.array(state)
        shape = (1,) + self._state_shape
        with torch.no_grad():
            q = self._net.get_q_values(torch.FloatTensor(state.reshape(shape)).to(self._device))
            return torch.argmax(q).item()

    def get_qvals(self, state):
        """Get Q-values for all actions in a state."""
        shape = (1,) + self._state_shape
        with torch.no_grad():
            q = self._net.get_q_values(torch.FloatTensor(state.reshape(shape)).to(self._device))
            return q

    def get_successor_features(self, state):
        """Get successor features for all actions in a state."""
        shape = (1,) + self._state_shape
        with torch.no_grad():
            sr, phi = self._net(torch.FloatTensor(state.reshape(shape)).to(self._device))
            return sr, phi

    def get_state_features(self, state):
        """Get state feature representation phi(s)."""
        shape = (1,) + self._state_shape
        with torch.no_grad():
            _, phi = self._net(torch.FloatTensor(state.reshape(shape)).to(self._device))
            return phi

    def step(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        new_state: tuple[int, int],
        active: bool,
    ):
        """Perform one step of training.

        Returns:
            dict with loss values (sr_loss, reward_loss, reconstruction_loss, total_loss)
        """
        self._buffer.push(state, action, reward, new_state, active)

        if len(self._buffer) < self._burnin:
            return {
                "loss": np.nan,
                "sr_loss": np.nan,
                "reward_loss": np.nan,
                "reconstruction_loss": np.nan,
            }

        states, actions, rewards, next_states, actives = self._buffer.sample(
            self._batch_size
        )

        shape = (self._batch_size,) + self._state_shape
        states_t = torch.FloatTensor(states.reshape(shape)).to(self._device)
        next_states_t = torch.FloatTensor(next_states.reshape(shape)).to(self._device)
        actions_t = torch.LongTensor(actions).to(self._device)
        rewards_t = torch.FloatTensor(rewards).to(self._device)
        actives_t = torch.FloatTensor(actives).to(self._device).clamp(0, 1)

        # Forward pass
        sr, phi = self._net(states_t)
        # sr: (batch, num_actions, feature_dim)
        # phi: (batch, feature_dim)

        # Get successor features for taken actions
        # sr_a: (batch, feature_dim)
        sr_a = sr.gather(1, actions_t.view(-1, 1, 1).expand(-1, 1, self._feature_dim)).squeeze(1)

        # Compute target successor features using target network
        with torch.no_grad():
            next_sr, next_phi = self._target_net(next_states_t)
            # For target, use greedy action selection
            next_q = torch.einsum('baf,f->ba', next_sr, self._net.reward_weights)
            next_actions = next_q.argmax(dim=1)
            # Get SR for greedy actions
            next_sr_a = next_sr.gather(1, next_actions.view(-1, 1, 1).expand(-1, 1, self._feature_dim)).squeeze(1)

            # TD target for SR: phi(s) + gamma * psi(s', a')
            sr_target = phi.detach() + self._discount_factor * next_sr_a * actives_t.unsqueeze(1)

        # Successor representation loss (MSE)
        sr_loss = nn.MSELoss()(sr_a, sr_target)

        # Reward prediction loss: r = phi(s) @ w
        predicted_rewards = torch.einsum('bf,f->b', phi, self._net.reward_weights)
        reward_loss = nn.MSELoss()(predicted_rewards, rewards_t)

        # Total loss
        total_loss = sr_loss + reward_loss

        # Optional reconstruction loss
        reconstruction_loss_value = 0.0
        if self._reconstruction:
            reconstructed = self._net.reconstruct(phi)
            reconstruction_loss = nn.MSELoss()(reconstructed, states_t)
            total_loss = total_loss + self._reconstruction_coef * reconstruction_loss
            reconstruction_loss_value = reconstruction_loss.item()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self._step_count += 1
        if self._step_count % self._target_update_frequency == 0:
            self._sync_target_network()

        self._exploration_rate = max(
            self._exploration_rate * self._exploration_decay, 0.01
        )

        return {
            "loss": total_loss.item(),
            "sr_loss": sr_loss.item(),
            "reward_loss": reward_loss.item(),
            "reconstruction_loss": reconstruction_loss_value,
        }

    def save_model(self, path, episode):
        """Save model checkpoint."""
        save_path = os.path.join(path, f"dsr_model_{episode}.pth")
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "step_count": self._step_count,
            },
            save_path,
        )

    def load_model(self, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self._device)
        self._net.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._step_count = checkpoint["step_count"]
        self._target_net.load_state_dict(self._net.state_dict())
