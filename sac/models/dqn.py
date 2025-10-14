from sac.models import base

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ConvDQNNet(nn.Module):
    def __init__(self, output_dim, optimistic_init=0.0):
        super(ConvDQNNet, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # Assuming single-channel input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)  # infers in_features on first forward
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

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
        return x


class FFDQNNet(nn.Module):
    def __init__(self, input_dim, output_dim, optimistic_init=0.0):
        super(FFDQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        # Optimistic initialization
        if optimistic_init > 0:
            nn.init.constant_(self.fc3.bias, optimistic_init)

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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


class DQN(base.BaseModel):

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
            convolutional: bool = False,
            optimistic_init: float = 0.0,
            weight_decay: float = 0.0,
        ):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if convolutional:
            net = ConvDQNNet(output_dim=num_actions, optimistic_init=optimistic_init)
            self._target_net = ConvDQNNet(output_dim=num_actions).to(self._device)
            self._state_shape = sample_state.shape[1:]  # cut batch dimension
        else:
            if isinstance(sample_state, tuple):
                sample_state = np.array(sample_state)
            net = FFDQNNet(input_dim=len(sample_state.flatten()), output_dim=num_actions, optimistic_init=optimistic_init)
            self._target_net = FFDQNNet(input_dim=len(sample_state.flatten()), output_dim=num_actions).to(self._device)
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
            q = self._net(torch.FloatTensor(state.reshape(shape)).to(self._device))
            return torch.argmax(q).item()
    
    def get_qvals(self, state):
        shape = (1,) + self._state_shape
        with torch.no_grad():
            q = self._net(torch.FloatTensor(state.reshape(shape)).to(self._device))
            return q
    
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
        values = self._net(torch.FloatTensor(states.reshape(shape)).to(self._device))
        state_action_values = values.gather(
            1, torch.LongTensor(actions).unsqueeze(1).to(self._device)
        ).squeeze()

        with torch.no_grad():
            next_values = self._target_net(
                torch.FloatTensor(next_states.reshape(shape)).to(self._device)
            )
            next_state_max_values = next_values.max(dim=1).values
            rewards = torch.FloatTensor(rewards).to(self._device)
            actives = torch.FloatTensor(actives).to(self._device)
            actives = actives.clamp(0, 1)  # Ensure binary values

            targets = rewards + self._discount_factor * next_state_max_values * actives

        targets = targets.detach()

        loss = nn.MSELoss()(state_action_values, targets)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._step_count += 1
        if self._step_count % self._target_update_frequency == 0:
            self._target_net.load_state_dict(self._net.state_dict())

        self._exploration_rate = max(
            self._exploration_rate * self._exploration_decay, 0.01
        )

        return {"loss": loss.item()}

    def save_model(self, path, episode):
        save_path = os.path.join(path, f"dqn_model_{episode}.pth")
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "step_count": self._step_count,
            },
            save_path,
        )
