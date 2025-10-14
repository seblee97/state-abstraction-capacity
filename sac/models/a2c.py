import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from sac.models import base


class ConvActorCriticNet(nn.Module):
    """Convolutional Actor-Critic Network for pixel-based observations"""
    def __init__(self, num_actions):
        super(ConvActorCriticNet, self).__init__()
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc_shared = nn.LazyLinear(128)
        
        # Actor head (policy)
        self.fc_actor = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, num_actions)
        
        # Critic head (value function)
        self.fc_critic = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x):
        # Shared layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_shared(x))
        
        # Actor (policy)
        actor = F.relu(self.fc_actor(x))
        policy_logits = self.policy_head(actor)
        
        # Critic (value)
        critic = F.relu(self.fc_critic(x))
        state_value = self.value_head(critic)
        
        return policy_logits, state_value


class FFActorCriticNet(nn.Module):
    """Feed-Forward Actor-Critic Network for position-based observations"""
    def __init__(self, input_dim, num_actions):
        super(FFActorCriticNet, self).__init__()
        # Shared layers
        self.fc_shared1 = nn.Linear(input_dim, 128)
        self.fc_shared2 = nn.Linear(128, 128)
        
        # Actor head (policy)
        self.fc_actor = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, num_actions)
        
        # Critic head (value function)
        self.fc_critic = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        # Shared layers
        x = F.relu(self.fc_shared1(x))
        x = F.relu(self.fc_shared2(x))
        
        # Actor (policy)
        actor = F.relu(self.fc_actor(x))
        policy_logits = self.policy_head(actor)
        
        # Critic (value)
        critic = F.relu(self.fc_critic(x))
        state_value = self.value_head(critic)
        
        return policy_logits, state_value


class A2C(base.BaseModel):
    """
    Advantage Actor-Critic (A2C) implementation
    
    A2C is a synchronous version of A3C that uses the advantage function
    to reduce variance in policy gradient updates.
    """
    
    def __init__(
        self,
        sample_state,
        num_actions,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        convolutional: bool = False,
        weight_decay: float = 0.0,
    ):
        """
        Initialize A2C model
        
        Args:
            sample_state: Sample state to determine input dimensions
            num_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            discount_factor: Gamma for discounting future rewards
            value_loss_coef: Coefficient for value loss in total loss
            entropy_coef: Coefficient for entropy bonus (encourages exploration)
            max_grad_norm: Maximum gradient norm for clipping
            convolutional: Whether to use convolutional architecture
            weight_decay: Weight decay (L2 regularization) factor
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if convolutional:
            self._net = ConvActorCriticNet(num_actions=num_actions).to(self._device)
            self._state_shape = sample_state.shape[1:]  # Cut batch dimension
        else:
            input_dim = len(sample_state)
            self._net = FFActorCriticNet(input_dim=input_dim, num_actions=num_actions).to(self._device)
            self._state_shape = (input_dim,)
        
        self._num_actions = num_actions
        self._discount_factor = discount_factor
        self._value_loss_coef = value_loss_coef
        self._entropy_coef = entropy_coef
        self._max_grad_norm = max_grad_norm

        self._optimizer = optim.AdamW(self._net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Buffers for storing trajectory data
        self._states = []
        self._actions = []
        self._rewards = []
        self._values = []
        self._log_probs = []
        self._dones = []
        
        self._step_count = 0
        
    def select_action(self, state):
        """Select action using current policy (stochastic)"""
        if isinstance(state, tuple):
            state = torch.tensor(state, dtype=torch.float32)
        shape = (1,) + self._state_shape
        state_tensor = torch.FloatTensor(state.reshape(shape)).to(self._device)
        
        with torch.no_grad():
            policy_logits, _ = self._net(state_tensor)
        
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        
        return action.item()
    
    def select_greedy_action(self, state):
        """Select action greedily (deterministic, for testing)"""
        if isinstance(state, tuple):
            state = torch.tensor(state, dtype=torch.float32)
        shape = (1,) + self._state_shape
        state_tensor = torch.FloatTensor(state.reshape(shape)).to(self._device)
        
        with torch.no_grad():
            policy_logits, _ = self._net(state_tensor)
        
        return torch.argmax(policy_logits).item()
    
    def step(
        self,
        state,
        action: int,
        reward: float,
        new_state,
        active: bool,
    ):
        """
        Store transition and perform update
        
        A2C updates after each step (online learning)
        """
        if isinstance(state, tuple):
            state = torch.tensor(state, dtype=torch.float32)
            new_state = torch.tensor(new_state, dtype=torch.float32)
        shape = (1,) + self._state_shape
        state_tensor = torch.FloatTensor(state.reshape(shape)).to(self._device)
        new_state_tensor = torch.FloatTensor(new_state.reshape(shape)).to(self._device)
        
        # Get current action probabilities and value
        policy_logits, state_value = self._net(state_tensor)
        dist = Categorical(logits=policy_logits)
        log_prob = dist.log_prob(torch.tensor(action).to(self._device))
        
        # Get next state value for bootstrap
        with torch.no_grad():
            _, next_state_value = self._net(new_state_tensor)
        
        # Calculate TD target and advantage
        if active:
            td_target = reward + self._discount_factor * next_state_value
        else:
            td_target = torch.tensor([[reward]], dtype=torch.float32).to(self._device)
        
        advantage = td_target - state_value
        
        # Calculate losses
        # Actor loss (policy gradient with advantage)
        actor_loss = -log_prob * advantage.detach()
        
        # Critic loss (MSE between value and TD target)
        critic_loss = F.mse_loss(state_value, td_target.detach())
        
        # Entropy bonus (encourages exploration)
        entropy = dist.entropy()
        
        # Total loss
        total_loss = (
            actor_loss + 
            self._value_loss_coef * critic_loss - 
            self._entropy_coef * entropy
        )
        
        # Optimization step
        self._optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self._net.parameters(), self._max_grad_norm)
        
        self._optimizer.step()
        
        self._step_count += 1
        
        return {
            "loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }
    
    def save_model(self, path, episode):
        """Save model checkpoint"""
        save_path = os.path.join(path, f"a2c_model_{episode}.pth")
        torch.save({
            'model_state_dict': self._net.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'step_count': self._step_count,
        }, save_path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self._device)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._step_count = checkpoint['step_count']
        self._net.eval()