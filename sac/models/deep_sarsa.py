from sac.models import base

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FFSARSANet(nn.Module):
    def __init__(self, input_dim, output_dim, optimistic_init=0.0):
        super(FFSARSANet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        if optimistic_init > 0:
            nn.init.constant_(self.fc3.bias, optimistic_init)

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ConvSARSANet(nn.Module):
    def __init__(self, output_dim, optimistic_init=0.0):
        super(ConvSARSANet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

        if optimistic_init > 0:
            nn.init.constant_(self.fc3.bias, optimistic_init)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepSARSA(base.BaseModel):
    """
    Online Deep SARSA without a replay buffer.

    On-policy consistency is maintained by caching the next action selected
    during step() and returning it from the subsequent select_action() call,
    so the action used for bootstrapping matches the action actually taken.
    """

    def __init__(
        self,
        sample_state,
        num_actions,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        exploration_decay: float,
        target_update_frequency: int = 0,
        convolutional: bool = False,
        optimistic_init: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if convolutional:
            self._net = ConvSARSANet(output_dim=num_actions, optimistic_init=optimistic_init).to(self._device)
            self._state_shape = sample_state.shape[1:]
        else:
            if isinstance(sample_state, tuple):
                sample_state = np.array(sample_state)
            input_dim = len(sample_state.flatten())
            self._net = FFSARSANet(input_dim=input_dim, output_dim=num_actions, optimistic_init=optimistic_init).to(self._device)
            self._state_shape = (input_dim,)

        # Optional target network (disabled by default: target_update_frequency=0)
        self._use_target = target_update_frequency > 0
        if self._use_target:
            if convolutional:
                self._target_net = ConvSARSANet(output_dim=num_actions).to(self._device)
            else:
                self._target_net = FFSARSANet(input_dim=self._state_shape[0], output_dim=num_actions).to(self._device)
            self._target_net.load_state_dict(self._net.state_dict())
            self._target_net.eval()
        else:
            self._target_net = None

        self._num_actions = num_actions
        self._optimizer = optim.AdamW(self._net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay
        self._target_update_frequency = target_update_frequency
        self._step_count = 0

        # Cache for on-policy action consistency
        self._cached_next_action = None

        super().__init__()

    def _state_tensor(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        shape = (1,) + self._state_shape
        return torch.FloatTensor(state.reshape(shape)).to(self._device)

    def _sample_epsilon_greedy(self, state):
        if np.random.rand() < self._exploration_rate:
            return np.random.choice(self._num_actions)
        return self.select_greedy_action(state)

    def select_action(self, state):
        # Return cached next action if available (ensures on-policy consistency)
        if self._cached_next_action is not None:
            action = self._cached_next_action
            self._cached_next_action = None
            return action
        return self._sample_epsilon_greedy(state)

    def select_greedy_action(self, state):
        with torch.no_grad():
            q = self._net(self._state_tensor(state))
            return torch.argmax(q).item()

    def get_qvals(self, state):
        with torch.no_grad():
            return self._net(self._state_tensor(state))

    def step(
        self,
        state,
        action: int,
        reward: float,
        new_state,
        active: bool,
    ):
        # Select next action from current policy and cache it for select_action()
        if active:
            next_action = self._sample_epsilon_greedy(new_state)
            self._cached_next_action = next_action
        else:
            next_action = None
            self._cached_next_action = None  # reset at episode boundary

        # Compute SARSA target
        with torch.no_grad():
            if active and next_action is not None:
                bootstrap_net = self._target_net if self._use_target else self._net
                next_q_values = bootstrap_net(self._state_tensor(new_state))
                next_q = next_q_values[0, next_action]
                target = reward + self._discount_factor * next_q
            else:
                target = torch.tensor(reward, dtype=torch.float32, device=self._device)

        target = target.detach() if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.float32, device=self._device)

        # Current Q(s, a)
        q_values = self._net(self._state_tensor(state))
        q_sa = q_values[0, action]

        loss = F.mse_loss(q_sa, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._step_count += 1
        if self._use_target and self._step_count % self._target_update_frequency == 0:
            self._target_net.load_state_dict(self._net.state_dict())

        self._exploration_rate = max(
            self._exploration_rate * self._exploration_decay, 0.01
        )

        return {"loss": loss.item()}

    def save_model(self, path, episode):
        save_path = os.path.join(path, f"deep_sarsa_model_{episode}.pth")
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "step_count": self._step_count,
            },
            save_path,
        )


class DeepSARSALambda(base.BaseModel):
    """
    Online Deep SARSA(λ) with eligibility traces. No replay buffer.

    Eligibility traces propagate the TD error back to all recently visited
    (state, action) pairs weighted by recency, dramatically accelerating
    credit assignment in long-horizon tasks compared to one-step SARSA.

    Because traces require applying the raw gradient ∂Q(s,a)/∂θ directly,
    we use manual SGD-style updates instead of Adam:
        e  ←  γλ · e  +  ∂Q(s,a)/∂θ
        θ  ←  θ  +  α · δ · e
    where δ = r + γ·Q(s',a') - Q(s,a) is the TD error.

    Traces are reset to zero at episode boundaries.
    """

    def __init__(
        self,
        sample_state,
        num_actions,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        exploration_decay: float,
        lambda_: float = 0.8,
        convolutional: bool = False,
        optimistic_init: float = 0.0,
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._convolutional = convolutional
        if convolutional:
            self._net = ConvSARSANet(output_dim=num_actions, optimistic_init=optimistic_init).to(self._device)
            self._state_shape = sample_state.shape[1:]
            # Trigger LazyLinear initialisation so all parameters exist before
            # we build the traces dict
            with torch.no_grad():
                self._net(torch.zeros(1, *self._state_shape, device=self._device))
        else:
            if isinstance(sample_state, tuple):
                sample_state = np.array(sample_state)
            input_dim = len(sample_state.flatten())
            self._net = FFSARSANet(input_dim=input_dim, output_dim=num_actions, optimistic_init=optimistic_init).to(self._device)
            self._state_shape = (input_dim,)

        self._num_actions = num_actions
        self._lr = learning_rate
        self._discount_factor = discount_factor
        self._lambda = lambda_
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay
        self._step_count = 0

        # Eligibility traces — one tensor per parameter, same shape
        self._traces = {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self._net.named_parameters()
        }

        # Cache for on-policy action consistency (same as DeepSARSA)
        self._cached_next_action = None

        super().__init__()

    def _state_tensor(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        state = state.astype(np.float32)
        if not self._convolutional:
            state = state / 37.0  # normalise (x,y) coords to [0,1]
        shape = (1,) + self._state_shape
        return torch.FloatTensor(state.reshape(shape)).to(self._device)

    def _sample_epsilon_greedy(self, state):
        if np.random.rand() < self._exploration_rate:
            return np.random.choice(self._num_actions)
        return self.select_greedy_action(state)

    def select_action(self, state):
        if self._cached_next_action is not None:
            action = self._cached_next_action
            self._cached_next_action = None
            return action
        return self._sample_epsilon_greedy(state)

    def select_greedy_action(self, state):
        with torch.no_grad():
            q = self._net(self._state_tensor(state))
            return torch.argmax(q).item()

    def get_qvals(self, state):
        with torch.no_grad():
            return self._net(self._state_tensor(state))

    def _reset_traces(self):
        for e in self._traces.values():
            e.zero_()

    def step(
        self,
        state,
        action: int,
        reward: float,
        new_state,
        active: bool,
    ):
        # Select and cache next action for on-policy consistency
        if active:
            next_action = self._sample_epsilon_greedy(new_state)
            self._cached_next_action = next_action
        else:
            next_action = None
            self._cached_next_action = None

        # Compute TD error δ = r + γQ(s',a') - Q(s,a)
        q_values = self._net(self._state_tensor(state))
        q_sa = q_values[0, action]

        with torch.no_grad():
            if active and next_action is not None:
                next_q = self._net(self._state_tensor(new_state))[0, next_action]
                td_target = reward + self._discount_factor * next_q
            else:
                td_target = torch.tensor(reward, dtype=torch.float32, device=self._device)
            delta = (td_target - q_sa).item()

        # Compute ∂Q(s,a)/∂θ
        self._net.zero_grad()
        q_sa.backward()

        # Update traces and apply parameter update
        with torch.no_grad():
            for name, param in self._net.named_parameters():
                if param.grad is not None:
                    # Accumulate: e ← γλ·e + ∂Q(s,a)/∂θ
                    self._traces[name].mul_(self._discount_factor * self._lambda)
                    self._traces[name].add_(param.grad)
                    self._traces[name].clamp_(-10.0, 10.0)
                    # SGD step: θ ← θ + α·δ·e
                    param.add_(self._lr * delta * self._traces[name])

        if not active:
            self._reset_traces()

        self._exploration_rate = max(
            self._exploration_rate * self._exploration_decay, 0.01
        )
        self._step_count += 1

        return {"loss": delta ** 2}

    def save_model(self, path, episode):
        save_path = os.path.join(path, f"deep_sarsa_lambda_model_{episode}.pth")
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "step_count": self._step_count,
            },
            save_path,
        )
