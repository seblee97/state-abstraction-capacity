import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from sac.models import base


class BaseActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def evaluate(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(dist.probs, dim=-1)
        return action, value


class ConvActorCritic(BaseActorCritic):

    def __init__(self, output_dim):
        super(ConvActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # Assuming single-channel input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)  # infers in_features on first forward
        self.fc2 = nn.Linear(128, 128)
        self.pi = nn.Linear(128, output_dim)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


class FFActorCritic(BaseActorCritic):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        hid = 64
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hid),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hid, output_dim)
        self.v = nn.Linear(hid, 1)

    def forward(self, x):
        x = self.fc(x)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


class RolloutBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0
        self.full = False

        # self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        # self.actions = torch.zeros((size,), dtype=torch.long, device=device)
        # self.logp = torch.zeros((size,), dtype=torch.float32, device=device)
        # self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        # self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        # self.values = torch.zeros((size,), dtype=torch.float32, device=device)

        # computed after trajectory collection
        self.adv = np.zeros((size,), dtype=np.float32)
        self.returns = np.zeros((size,), dtype=np.float32)

    def push(self, state, action, reward, active, logp, value):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, active, logp, value)

        # self.obs[self.ptr] = torch.FloatTensor(obs).to(self._device)
        # self.actions[self.ptr] = torch.LongTensor(action).to(self._device)
        # self.logp[self.ptr] = torch.FloatTensor(logp).to(self._device)
        # self.rewards[self.ptr] = torch.FloatTensor(reward).to(self._device)
        # self.dones[self.ptr] = torch.FloatTensor(done).to(self._device)
        # self.values[self.ptr] = torch.FloatTensor(value).to(self._device)
        self.position += 1
        if self.position >= self.size:
            self.full = True

    def reset(self):
        self.position = 0
        self.full = False

    def compute_gae(self, last_value, gamma, lam):
        actual_size = self.position if not self.full else self.size
        adv = 0.0
        for t in reversed(range(actual_size)):
            next_nonterminal = self.buffer[t][3]
            next_value = last_value if t == actual_size - 1 else self.buffer[t + 1][5]
            delta = (
                self.buffer[t][2]
                + gamma * next_value * next_nonterminal
                - self.buffer[t][5]
            )
            adv = delta + gamma * lam * next_nonterminal * adv
            self.adv[t] = adv
        self.returns[:actual_size] = np.array(self.adv[:actual_size]) + np.array([
            i[5] for i in self.buffer[:actual_size]
        ]).flatten()
        # Normalize advantages only over actual data
        if actual_size > 1:
            self.adv[:actual_size] = (
                self.adv[:actual_size] - self.adv[:actual_size].mean()
            ) / (self.adv[:actual_size].std() + 1e-8)

    def get_minibatches(self, batch_size, shuffle=True):
        idxs = np.arange(self.size)
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            mb_idx = idxs[start:end]
            states, actions, rewards, actives, logps, values = zip(
                *[self.buffer[idx] for idx in mb_idx]
            )
            yield (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(actives),
                np.array(logps),
                np.array(values),
                self.adv[mb_idx],
                self.returns[mb_idx],
            )


class PPO(base.BaseModel):

    def __init__(
        self,
        sample_state,
        num_actions,
        batch_size: int,
        replay_buffer_size: int,
        learning_rate: float,
        discount_factor: float,
        gae_lambda: float,
        clip_coef: float,
        vf_coef: float,
        ent_coef: float,
        max_grad_norm: float,
        convolutional: bool = False,
    ):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if convolutional:
            net = ConvActorCritic(output_dim=num_actions)
            self._state_shape = sample_state.shape[1:]  # cut batch dimension
        else:
            net = FFActorCritic(
                input_dim=len(sample_state.flatten()), output_dim=num_actions
            )
            self._state_shape = (len(sample_state.flatten()),)

        self._net = net.to(self._device)

        self._num_actions = num_actions

        self._optimizer = optim.Adam(self._net.parameters(), lr=learning_rate)

        self._buffer = RolloutBuffer(size=replay_buffer_size)

        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._gae_lambda = gae_lambda
        self._clip_coef = clip_coef
        self._vf_coef = vf_coef
        self._ent_coef = ent_coef
        self._max_grad_norm = max_grad_norm

        self._step_count = 0

        super().__init__()

    def select_action(self, state):
        shape = (1,) + self._state_shape
        return self._net.act(
            torch.FloatTensor(state.reshape(shape)).to(self._device)
        )

    def select_greedy_action(self, state):
        shape = (1,) + self._state_shape
        return self._net.evaluate(
            torch.FloatTensor(state.reshape(shape)).to(self._device)
        )[0].item()

    def forward(self, state):
        shape = (1,) + self._state_shape
        return self._net(
            torch.FloatTensor(state.reshape(shape)).to(self._device)
        )

    def compute_gae(self, last_value):
        self._buffer.compute_gae(last_value, self._discount_factor, self._gae_lambda)

    def add_to_buffer(self, state, action, logp, reward, active, value):
        self._buffer.push(state, action, logp, reward, active, value)

    def step(self):

        losses = []
        policy_losses = []
        value_losses = []
        entropies = []

        for (
            states,
            actions,
            rewards,
            actives,
            old_logps,
            values,
            advantages,
            returns,
        ) in self._buffer.get_minibatches(self._batch_size):

            shape = (states.shape[0],) + self._state_shape
            logits, values_pred = self._net(
                torch.FloatTensor(states.reshape(shape)).to(self._device)
            )
            dist = Categorical(logits=logits)
            logp = dist.log_prob(torch.LongTensor(actions).to(self._device))
            entropy = dist.entropy().mean()

            # Convert to tensors and move to device
            old_logps_tensor = torch.FloatTensor(old_logps).to(self._device)
            advantages_tensor = torch.FloatTensor(advantages).to(self._device)
            returns_tensor = torch.FloatTensor(returns).to(self._device)

            ratio = (logp - old_logps_tensor).exp()
            unclipped = ratio * advantages_tensor
            clipped = (
                torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef)
                * advantages_tensor
            )
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = 0.5 * (returns_tensor - values_pred).pow(2).mean()

            loss = policy_loss + self._vf_coef * value_loss - self._ent_coef * entropy

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), self._max_grad_norm)
            self._optimizer.step()

            self._step_count += 1

            losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

        return {
            "loss": np.mean(losses),
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }

    def save_model(self, path: str, step: int) -> None:
        save_path = os.path.join(path, f"dqn_model_{step}.pth")
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "step_count": self._step_count,
            },
            save_path,
        )

    def reset_buffer(self):
        self._buffer.reset()
