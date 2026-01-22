from sac.models import base

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ConvQRDQNNet(nn.Module):
    def __init__(self, num_actions, num_quantiles=200, optimistic_init=0.0):
        super(ConvQRDQNNet, self).__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # Assuming single-channel input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)  # infers in_features on first forward
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions * num_quantiles)

        # Optimistic initialization
        if optimistic_init > 0:
            nn.init.constant_(self.fc3.bias, optimistic_init)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape to (batch_size, num_actions, num_quantiles)
        x = x.view(-1, self.num_actions, self.num_quantiles)
        return x


class FFQRDQNNet(nn.Module):
    def __init__(self, input_dim, num_actions, num_quantiles=200, optimistic_init=0.0):
        super(FFQRDQNNet, self).__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions * num_quantiles)

        # Optimistic initialization
        if optimistic_init > 0:
            nn.init.constant_(self.fc3.bias, optimistic_init)

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape to (batch_size, num_actions, num_quantiles)
        x = x.view(-1, self.num_actions, self.num_quantiles)
        return x


class ReplayBuffer:
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


def quantile_huber_loss(quantiles, target, actions, kappa=1.0):
    """
    Compute the quantile Huber loss for QRDQN.

    Args:
        quantiles: (batch_size, num_actions, num_quantiles) - predicted quantiles
        target: (batch_size, num_quantiles) - target quantiles
        actions: (batch_size,) - actions taken
        kappa: threshold for Huber loss

    Returns:
        loss: scalar loss value
    """
    batch_size = quantiles.size(0)
    num_quantiles = quantiles.size(2)

    # Get quantiles for the actions taken: (batch_size, num_quantiles)
    quantiles = quantiles[torch.arange(batch_size), actions]

    # Expand dimensions for broadcasting
    # quantiles: (batch_size, num_quantiles, 1)
    # target: (batch_size, 1, num_quantiles)
    quantiles = quantiles.unsqueeze(2)
    target = target.unsqueeze(1)

    # Compute TD errors: (batch_size, num_quantiles, num_quantiles)
    td_errors = target - quantiles

    # Huber loss
    huber_loss = torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa)
    )

    # Quantile regression loss
    tau = torch.arange(0, num_quantiles, dtype=torch.float32, device=quantiles.device)
    tau = (tau + 0.5) / num_quantiles  # midpoint of each quantile
    tau = tau.view(1, num_quantiles, 1)

    quantile_loss = torch.abs(tau - (td_errors < 0).float()) * huber_loss

    # Standard QRDQN loss: sum over target quantiles, mean over predicted quantiles and batch
    loss = quantile_loss.sum(dim=2).mean(dim=1).mean(dim=0)

    return loss


class QRDQN(base.BaseModel):

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
            num_quantiles: int = 200,
            kappa: float = 1.0,
            convolutional: bool = False,
            optimistic_init: float = 0.0,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
        ):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._num_quantiles = num_quantiles
        self._kappa = kappa

        if convolutional:
            net = ConvQRDQNNet(
                num_actions=num_actions,
                num_quantiles=num_quantiles,
                optimistic_init=optimistic_init
            )
            self._target_net = ConvQRDQNNet(
                num_actions=num_actions,
                num_quantiles=num_quantiles
            ).to(self._device)
            self._state_shape = sample_state.shape[1:]  # cut batch dimension
        else:
            if isinstance(sample_state, tuple):
                sample_state = np.array(sample_state)
            net = FFQRDQNNet(
                input_dim=len(sample_state.flatten()),
                num_actions=num_actions,
                num_quantiles=num_quantiles,
                optimistic_init=optimistic_init
            )
            self._target_net = FFQRDQNNet(
                input_dim=len(sample_state.flatten()),
                num_actions=num_actions,
                num_quantiles=num_quantiles
            ).to(self._device)
            self._state_shape = (len(sample_state.flatten()),)

        self._net = net.to(self._device)

        self._target_net.load_state_dict(self._net.state_dict())
        self._target_net.eval()

        self._num_actions = num_actions

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
        self._max_grad_norm = max_grad_norm

        super().__init__()

    def select_action(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        if np.random.rand() < self._exploration_rate:
            action = np.random.choice(range(self._num_actions))
        else:
            action = self.select_greedy_action(state)
        return action

    def select_greedy_action(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        shape = (1,) + self._state_shape
        with torch.no_grad():
            quantiles = self._net(torch.FloatTensor(state.reshape(shape)).to(self._device))
            # Average over quantiles to get Q-values: (1, num_actions)
            q_values = quantiles.mean(dim=2)
            return torch.argmax(q_values).item()

    def get_qvals(self, state):
        """Get Q-values by averaging over quantiles."""
        shape = (1,) + self._state_shape
        with torch.no_grad():
            quantiles = self._net(torch.FloatTensor(state.reshape(shape)).to(self._device))
            # Average over quantiles to get Q-values
            q_values = quantiles.mean(dim=2)
            return q_values

    def step(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        new_state: tuple[int, int],
        active: bool,
    ):

        self._buffer.push(state, action, reward, new_state, active)

        if len(self._buffer) < self._burnin:
            return {"loss": np.nan}  # Return dict instead of None

        states, actions, rewards, next_states, actives = self._buffer.sample(
            self._batch_size
        )

        shape = (self._batch_size,) + self._state_shape

        # Get quantiles for current states: (batch_size, num_actions, num_quantiles)
        quantiles = self._net(torch.FloatTensor(states.reshape(shape)).to(self._device))

        with torch.no_grad():
            # Get next state quantiles from target network
            next_quantiles_target = self._target_net(
                torch.FloatTensor(next_states.reshape(shape)).to(self._device)
            )

            # Select best action based on mean Q-value from target network
            next_q_values = next_quantiles_target.mean(dim=2)
            next_actions = next_q_values.argmax(dim=1)

            # Get quantiles for best actions: (batch_size, num_quantiles)
            next_quantiles = next_quantiles_target[torch.arange(self._batch_size), next_actions]

            rewards = torch.FloatTensor(rewards).to(self._device)
            actives = torch.FloatTensor(actives).to(self._device)
            actives = actives.clamp(0, 1)  # Ensure binary values

            # Compute target quantiles: (batch_size, num_quantiles)
            target_quantiles = rewards.unsqueeze(1) + self._discount_factor * next_quantiles * actives.unsqueeze(1)

        target_quantiles = target_quantiles.detach()

        # Compute quantile Huber loss
        actions_tensor = torch.LongTensor(actions).to(self._device)
        loss = quantile_huber_loss(quantiles, target_quantiles, actions_tensor, kappa=self._kappa)

        self._optimizer.zero_grad()
        loss.backward()
        if self._max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), self._max_grad_norm)
        self._optimizer.step()

        self._step_count += 1
        if self._step_count % self._target_update_frequency == 0:
            self._target_net.load_state_dict(self._net.state_dict())

        self._exploration_rate = max(
            self._exploration_rate * self._exploration_decay, 0.01
        )

        return {"loss": loss.item()}

    def save_model(self, path, episode):
        save_path = os.path.join(path, f"qrdqn_model_{episode}.pth")
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "step_count": self._step_count,
            },
            save_path,
        )
