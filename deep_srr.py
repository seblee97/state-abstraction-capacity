"""
Deep Successor Representation for GridWorld
============================================

Requirements:
    pip install torch gymnasium numpy
    pip install gridworld_env

Usage:
    python deep_sr.py
"""

import random
import json
import csv
import argparse
import dataclasses
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gridworld_env import GridWorldEnv
import gridworld_env


# =============================================================================
# HYPERPARAMETERS - All configurable parameters in one place
# =============================================================================


@dataclass
class Config:
    # Environment
    layout_name: str = "meister.txt"
    max_steps_per_episode: int = 500
    step_reward: float = 0.0
    collision_reward: float = -0.1
    random_start: bool = True  # Random start position during training

    # Training
    n_episodes: int = 10000
    print_every: int = 1
    seed: int = 42  # Random seed for reproducibility (None for no seeding)

    # Testing (greedy evaluation during training)
    test_every: int = 10  # Run test episodes every N training episodes
    n_test_episodes: int = 1  # Number of test episodes to run

    # Visualization
    render_every: int = 10  # Render episode every N episodes (0 to disable)
    render_test: bool = True  # Also render test episodes
    render_fps: int = 10
    save_video: bool = True  # Save as video file vs display live

    # Output (logs and videos go here)
    output_dir: str = "/mnt/home/slee1/ceph/sacmeister"
    experiment_name: str = "dsr"

    # Network
    feature_dim: int = 128

    # Learning
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128

    # Exploration (epsilon decays per episode)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.999  # per episode

    # Loss weights
    use_reconstruction: bool = True  # Whether to use reconstruction loss
    sr_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.1

    # Target network
    target_update_freq: int = 1000  # steps

    # Replay buffer
    buffer_capacity: int = 100000

    # Checkpointing
    checkpoint_every: int = 100  # Save checkpoint every N episodes (0 to disable)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# REPLAY BUFFER
# =============================================================================


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# NETWORK
# =============================================================================


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for grid observations.

    Architecture:
        Conv2d(1, 16, 3x3, padding=1) -> ReLU ->
        Conv2d(16, 32, 3x3, padding=1) -> ReLU ->
        Flatten -> LazyLinear(128) -> ReLU ->
        Linear(128, 128) -> ReLU -> Linear(128, output_dim)
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for state reconstruction.

    Mirrors the encoder architecture.
    """

    def __init__(self, input_dim: int, grid_height: int, grid_width: int):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Reverse of encoder FC layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32 * grid_height * grid_width)

        # Reverse of encoder conv layers
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 32, self.grid_height, self.grid_width)
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)  # No activation - output is reconstructed state
        return x.squeeze(1)  # (batch, height, width)


class DeepSuccessorRepresentation(nn.Module):
    """
    Deep Successor Representation network.

    Components:
        - Encoder phi(s): state -> features
        - Decoder (optional): features -> reconstructed state
        - SR heads psi(s,a): features -> successor features per action
        - Reward weights w: features -> reward

    Q(s, a) = psi(s, a) @ w
    """

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        grid_height: int,
        grid_width: int,
        use_reconstruction: bool = True,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.use_reconstruction = use_reconstruction

        self.encoder = ConvEncoder(output_dim=feature_dim)

        if use_reconstruction:
            self.decoder = ConvDecoder(
                input_dim=feature_dim, grid_height=grid_height, grid_width=grid_width
            )
        else:
            self.decoder = None

        self.sr_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, feature_dim)
                )
                for _ in range(n_actions)
            ]
        )

        self.reward_weights = nn.Linear(feature_dim, 1, bias=False)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, phi: torch.Tensor) -> Optional[torch.Tensor]:
        if self.decoder is not None:
            return self.decoder(phi)
        return None

    def reconstruct(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode then decode for reconstruction."""
        if self.decoder is not None:
            phi = self.encode(state)
            return self.decode(phi)
        return None

    def successor_features(self, state: torch.Tensor) -> torch.Tensor:
        phi = self.encode(state)
        psi_list = [head(phi) for head in self.sr_heads]
        return torch.stack(psi_list, dim=1)

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        psi = self.successor_features(state)
        return self.reward_weights(psi).squeeze(-1)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns phi, psi, q, reconstructed_state (or None if no decoder)."""
        phi = self.encode(state)
        psi = torch.stack([head(phi) for head in self.sr_heads], dim=1)
        q = self.reward_weights(psi).squeeze(-1)
        reconstructed = self.decode(phi)
        return phi, psi, q, reconstructed


# =============================================================================
# AGENT
# =============================================================================


class DSRAgent:
    def __init__(
        self, n_actions: int, config: Config, grid_height: int, grid_width: int
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.n_actions = n_actions
        self.epsilon = config.epsilon_start

        self.network = DeepSuccessorRepresentation(
            n_actions,
            config.feature_dim,
            grid_height,
            grid_width,
            use_reconstruction=config.use_reconstruction,
        ).to(self.device)
        self.target_network = DeepSuccessorRepresentation(
            n_actions,
            config.feature_dim,
            grid_height,
            grid_width,
            use_reconstruction=config.use_reconstruction,
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        # Target network should not be trained directly
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)
        self.update_count = 0
        self._initialized = False

    def _ensure_initialized(self, state: np.ndarray):
        if not self._initialized:
            with torch.no_grad():
                dummy = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                self.network(dummy)
                self.target_network(dummy)
                self.target_network.load_state_dict(self.network.state_dict())
            self._initialized = True

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        self._ensure_initialized(state)

        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network.q_values(state_tensor)
        self.network.train()
        return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(
            self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[dict]:
        cfg = self.config

        if len(self.replay_buffer) < cfg.batch_size:
            return None

        batch = self.replay_buffer.sample(cfg.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(
            self.device
        )
        dones = torch.FloatTensor([float(t.done) for t in batch]).to(self.device)

        # Forward pass through online network
        phi, psi, _, reconstructed = self.network(states)
        batch_indices = torch.arange(len(batch), device=self.device)
        psi_sa = psi[batch_indices, actions]

        # phi_next is features of next_states
        phi_next, _, _, _ = self.network(next_states)
        phi_next_online = phi_next  # <-- keep grad for reward loss
        phi_next = phi_next.detach()  # <-- detached for SR target

        with torch.no_grad():
            psi_next = self.target_network.successor_features(next_states)  # [B, A, d]
            q_next = self.target_network.q_values(next_states)  # [B, A]
            next_actions = q_next.argmax(dim=1)  # [B]
            psi_next_sa = psi_next[batch_indices, next_actions]  # [B, d]

        done = dones.unsqueeze(1).float()  # [B, 1]

        # shifted Bellman:
        # nonterminal: phi_next + gamma * psi_next_sa
        # terminal:    phi_next
        sr_target = phi_next + (1.0 - done) * (cfg.gamma * psi_next_sa)

        # SR Loss: MSE between predicted psi(s,a) and target
        sr_loss = F.mse_loss(psi_sa, sr_target)

        r_pred = self.network.reward_weights(
            phi_next_online
        )  # or reward_weights(phi_next)
        reward_loss = F.mse_loss(r_pred.squeeze(-1), rewards)

        total_loss = cfg.sr_loss_weight * sr_loss + cfg.reward_loss_weight * reward_loss

        # Optional reconstruction loss (also backprops through phi)
        reconstruction_loss = None
        if cfg.use_reconstruction and reconstructed is not None:
            reconstruction_loss = F.mse_loss(reconstructed, states)
            total_loss = (
                total_loss + cfg.reconstruction_loss_weight * reconstruction_loss
            )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % cfg.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        result = {
            "sr_loss": sr_loss.item(),
            "reward_loss": reward_loss.item(),
        }
        if reconstruction_loss is not None:
            result["reconstruction_loss"] = reconstruction_loss.item()

        return result

    def save_checkpoint(self, filepath: Path, episode: int, total_steps: int):
        """Save a checkpoint of the agent state."""
        checkpoint = {
            "episode": episode,
            "total_steps": total_steps,
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "config": asdict(self.config),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path) -> dict:
        """Load a checkpoint and return metadata (episode, total_steps)."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self._initialized = True
        return {
            "episode": checkpoint["episode"],
            "total_steps": checkpoint["total_steps"],
        }


# =============================================================================
# VISUALIZATION
# =============================================================================


def render_episode(
    agent,
    config: Config,
    episode: int,
    layout_path: Path,
    video_dir: Path,
    mode: str = "train",
):
    """
    Render a single episode using greedy policy.

    Args:
        agent: The DSR agent
        config: Config object
        episode: Current episode number
        layout_path: Path to layout file
        video_dir: Directory to save videos
        mode: "train" or "test" - used for labeling the video
    """
    import time

    render_mode = "rgb_array" if config.save_video else "human"
    render_env = GridWorldEnv(
        layout=str(layout_path),
        max_steps=config.max_steps_per_episode,
        step_reward=config.step_reward,
        collision_reward=config.collision_reward,
        flatten_obs=False,
        render_mode=render_mode,
    )

    frames = []
    obs, _ = render_env.reset()
    state = obs["grid"].astype(np.float32) / 8.0

    frame = render_env.render()
    if config.save_video:
        frames.append(frame)

    total_reward = 0
    steps = 0

    for _ in range(config.max_steps_per_episode):
        action = agent.select_action(state, greedy=mode == "test")
        next_obs, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        steps += 1

        frame = render_env.render()
        if config.save_video:
            frames.append(frame)
        else:
            time.sleep(1.0 / config.render_fps)

        state = next_obs["grid"].astype(np.float32) / 8.0

        if terminated or truncated:
            break

    render_env.close()

    if config.save_video and frames:
        save_video(
            frames, video_dir, episode, total_reward, steps, mode, config.render_fps
        )

    return total_reward, steps


def save_video(
    frames: List[np.ndarray],
    video_dir: Path,
    episode: int,
    reward: float,
    steps: int,
    mode: str,
    fps: int,
):
    """Save frames as video file."""
    try:
        import imageio
    except ImportError:
        print(
            "    Warning: imageio not installed, skipping video save. Install with: pip install imageio"
        )
        return

    # Try MP4 first, fall back to GIF if ffmpeg not available
    mp4_filename = video_dir / f"{mode}_ep{episode:04d}_r{reward:.1f}_s{steps}.mp4"
    gif_filename = video_dir / f"{mode}_ep{episode:04d}_r{reward:.1f}_s{steps}.gif"

    try:
        # Try to save as MP4 (requires ffmpeg)
        imageio.mimsave(str(mp4_filename), frames, fps=fps, codec="libx264")
        print(f"    Saved: {mp4_filename}")
    except Exception:
        try:
            # Fall back to GIF
            imageio.mimsave(str(gif_filename), frames, fps=fps)
            print(f"    Saved: {gif_filename}")
        except Exception as e:
            print(f"    Warning: Could not save video: {e}")
            print(
                "    Install ffmpeg for MP4 support: conda install ffmpeg OR brew install ffmpeg"
            )


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================


def get_valid_positions(env) -> List[Tuple[int, int]]:
    """Get positions that are non-wall and have more than 2 non-wall neighbors."""
    valid = []
    layout = env._base_layout
    for row in range(layout.height):
        for col in range(layout.width):
            if not layout.is_wall(row, col):
                # Count non-wall neighbors (up, down, left, right)
                neighbors = [
                    (row - 1, col),  # up
                    (row + 1, col),  # down
                    (row, col - 1),  # left
                    (row, col + 1),  # right
                ]
                non_wall_neighbors = sum(
                    1 for r, c in neighbors if not layout.is_wall(r, c)
                )
                if non_wall_neighbors > 2:
                    valid.append((row, col))
    return valid


def reset_with_random_start(env, valid_positions: List[Tuple[int, int]]) -> dict:
    """Reset environment and randomize agent start position."""
    obs, info = env.reset()
    # Include original start position in sampling pool
    start_pos = env._base_layout.start_position
    positions_to_sample = valid_positions + [start_pos]
    # Override agent position with random position
    new_pos = random.choice(positions_to_sample)
    env._agent_pos = new_pos
    # Rebuild observation with new position
    obs = env._get_observation()
    return obs, info


# =============================================================================
# LOGGING
# =============================================================================


class Logger:
    """Logs training statistics to CSV files."""

    def __init__(self, config: Config):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.output_dir) / f"{config.experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create videos subdirectory
        self.video_dir = self.run_dir / "videos"
        self.video_dir.mkdir(exist_ok=True)

        # Create checkpoints subdirectory
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Save config
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Initialize CSV files
        self.train_file = self.run_dir / "train.csv"
        self.test_file = self.run_dir / "test.csv"
        self.loss_file = self.run_dir / "loss.csv"

        # Write headers
        self._write_row(self.train_file, ["episode", "reward", "length", "epsilon"])
        self._write_row(
            self.test_file,
            ["episode", "mean_reward", "std_reward", "mean_length", "std_length"],
        )
        self._write_row(
            self.loss_file,
            ["episode", "step", "sr_loss", "reward_loss", "reconstruction_loss"],
        )

        print(f"Output directory: {self.run_dir}")

    def _write_row(self, filepath: Path, row: List):
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_train(self, episode: int, reward: float, length: int, epsilon: float):
        self._write_row(self.train_file, [episode, reward, length, epsilon])

    def log_test(
        self, episode: int, mean_r: float, std_r: float, mean_l: float, std_l: float
    ):
        self._write_row(self.test_file, [episode, mean_r, std_r, mean_l, std_l])

    def log_loss(
        self,
        episode: int,
        step: int,
        sr_loss: float,
        reward_loss: float,
        reconstruction_loss: float,
    ):
        self._write_row(
            self.loss_file, [episode, step, sr_loss, reward_loss, reconstruction_loss]
        )


# =============================================================================
# TESTING
# =============================================================================


def test_agent(
    agent,
    config: Config,
    layout_path: Path,
    episode: int,
    video_dir: Path,
    render: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Run test episodes with greedy policy.

    Returns:
        mean_reward, std_reward, mean_length, std_length
    """
    env = GridWorldEnv(
        layout=str(layout_path),
        max_steps=config.max_steps_per_episode,
        step_reward=config.step_reward,
        collision_reward=config.collision_reward,
        flatten_obs=False,
    )

    test_rewards = []
    test_lengths = []

    for i in range(config.n_test_episodes):
        obs, _ = env.reset()
        state = obs["grid"].astype(np.float32) / 8.0
        ep_reward = 0
        ep_length = 0

        for _ in range(config.max_steps_per_episode):
            action = agent.select_action(state, greedy=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_length += 1
            state = next_obs["grid"].astype(np.float32) / 8.0

            if terminated or truncated:
                break

        test_rewards.append(ep_reward)
        test_lengths.append(ep_length)

    env.close()

    # Render one test episode if requested
    if render and config.render_test:
        render_episode(agent, config, episode, layout_path, video_dir, mode="test")

    return (
        np.mean(test_rewards),
        np.std(test_rewards),
        np.mean(test_lengths),
        np.std(test_lengths),
    )


# =============================================================================
# TRAINING
# =============================================================================


def run_dsr_training(config: Config):
    print("=" * 60)
    print("Deep Successor Representation Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Config: {config}\n")

    # Set random seeds for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    # Find layout from installed package
    package_dir = Path(gridworld_env.__file__).parent
    layout_path = package_dir / "layouts" / config.layout_name

    if not layout_path.exists():
        raise FileNotFoundError(f"Layout not found: {layout_path}")

    # Create environment
    env = GridWorldEnv(
        layout=str(layout_path),
        max_steps=config.max_steps_per_episode,
        step_reward=config.step_reward,
        collision_reward=config.collision_reward,
        flatten_obs=False,
    )

    print(f"Grid size: {env._base_layout.height} x {env._base_layout.width}")
    print(f"Rewards: {len(env._base_layout.rewards)}")

    # Get valid positions for random start
    valid_positions = get_valid_positions(env)
    print(f"Valid positions: {len(valid_positions)}")
    if config.random_start:
        print("Using random start positions for training")

    # Initialize
    grid_height = env._base_layout.height
    grid_width = env._base_layout.width
    agent = DSRAgent(
        n_actions=4, config=config, grid_height=grid_height, grid_width=grid_width
    )
    logger = Logger(config)

    # Tracking
    train_rewards = []
    train_lengths = []
    test_results = []
    total_steps = 0

    for episode in range(config.n_episodes):
        # Reset with optional random start
        if config.random_start:
            obs, _ = reset_with_random_start(env, valid_positions)
        else:
            obs, _ = env.reset()
        state = obs["grid"].astype(np.float32) / 8.0

        ep_reward = 0
        ep_length = 0
        ep_losses = []

        for _ in range(config.max_steps_per_episode):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            # done = terminated or truncated

            episode_done = terminated or truncated  # for loop control
            bootstrap_done = terminated  # for learning targets

            next_state = next_obs["grid"].astype(np.float32) / 8.0

            agent.store_transition(state, action, reward, next_state, bootstrap_done)

            loss_info = agent.update()
            if loss_info:
                ep_losses.append(loss_info)
                recon_loss = loss_info.get("reconstruction_loss", 0.0)
                logger.log_loss(
                    episode + 1,
                    total_steps,
                    loss_info["sr_loss"],
                    loss_info["reward_loss"],
                    recon_loss,
                )

            ep_reward += reward
            ep_length += 1
            total_steps += 1
            state = next_state

            if episode_done:
                break

        agent.decay_epsilon()
        train_rewards.append(ep_reward)
        train_lengths.append(ep_length)

        # Log training episode
        logger.log_train(episode + 1, ep_reward, ep_length, agent.epsilon)

        # Print training progress
        if (episode + 1) % config.print_every == 0:
            avg_r = np.mean(train_rewards[-config.print_every :])
            avg_l = np.mean(train_lengths[-config.print_every :])
            loss_str = ""
            if ep_losses:
                avg_sr = np.mean([l["sr_loss"] for l in ep_losses])
                loss_str = f"SR: {avg_sr:.4f}, "
                if config.use_reconstruction and "reconstruction_loss" in ep_losses[0]:
                    avg_recon = np.mean([l["reconstruction_loss"] for l in ep_losses])
                    loss_str += f"Rec: {avg_recon:.4f}, "
            print(
                f"Ep {episode+1}/{config.n_episodes} | R: {avg_r:.2f} | L: {avg_l:.1f} | {loss_str}Eps: {agent.epsilon:.3f}"
            )

        # Render training episode periodically
        if config.render_every > 0 and (episode + 1) % config.render_every == 0:
            print(f"  Rendering training episode {episode + 1}...")
            r, s = render_episode(
                agent, config, episode + 1, layout_path, logger.video_dir, mode="train"
            )
            print(f"    Result: reward={r:.2f}, steps={s}")

        # Run test episodes periodically
        if config.test_every > 0 and (episode + 1) % config.test_every == 0:
            should_render = (
                config.render_every > 0 and (episode + 1) % config.render_every == 0
            )
            mean_r, std_r, mean_l, std_l = test_agent(
                agent,
                config,
                layout_path,
                episode + 1,
                logger.video_dir,
                render=should_render,
            )
            test_results.append((episode + 1, mean_r, std_r, mean_l, std_l))
            logger.log_test(episode + 1, mean_r, std_r, mean_l, std_l)
            print(
                f"  TEST @ {episode+1}: R={mean_r:.2f}±{std_r:.2f}, L={mean_l:.1f}±{std_l:.1f}"
            )

        # Save checkpoint periodically
        if config.checkpoint_every > 0 and (episode + 1) % config.checkpoint_every == 0:
            checkpoint_path = logger.checkpoint_dir / f"checkpoint_ep{episode+1:06d}.pt"
            agent.save_checkpoint(checkpoint_path, episode + 1, total_steps)
            print(f"  Checkpoint saved: {checkpoint_path}")

    env.close()

    # Save final checkpoint
    final_checkpoint_path = logger.checkpoint_dir / "checkpoint_final.pt"
    agent.save_checkpoint(final_checkpoint_path, config.n_episodes, total_steps)
    print(f"\nFinal checkpoint saved: {final_checkpoint_path}")
    print("Training complete!")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    mean_r, std_r, mean_l, std_l = test_agent(
        agent, config, layout_path, config.n_episodes, logger.video_dir, render=True
    )
    logger.log_test(config.n_episodes, mean_r, std_r, mean_l, std_l)
    print(
        f"Final test ({config.n_test_episodes} episodes): R={mean_r:.2f}±{std_r:.2f}, L={mean_l:.1f}±{std_l:.1f}"
    )

    print(f"\nAll outputs saved to: {logger.run_dir}")

    return agent, train_rewards, train_lengths, test_results


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> Config:
    """Build CLI parser from Config dataclass fields."""

    parser = argparse.ArgumentParser(
        description="Deep Successor Representation for GridWorld",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    for field in dataclasses.fields(Config):
        name = field.name
        ftype = field.type
        default = field.default

        if ftype == "bool" or ftype is bool:
            # --flag / --no-flag
            parser.add_argument(
                f"--{name}", action="store_true", default=None,
            )
            parser.add_argument(
                f"--no-{name}", dest=name, action="store_false",
            )
        else:
            origin = getattr(ftype, "__origin__", None)
            if isinstance(ftype, str):
                # Resolve string annotations
                ftype = eval(ftype)
            parser.add_argument(f"--{name}", type=ftype, default=None)

    args = parser.parse_args()

    # Only pass explicitly provided values to Config
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return Config(**overrides)


if __name__ == "__main__":
    config = parse_args()
    agent, train_rewards, train_lengths, test_results = run_dsr_training(config)
