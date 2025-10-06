from sac.models import q_learning, ppo, dqn, a2c
from key_door import key_door_env, visualisation_env
import argparse
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(
    description="Train RL models on the Key-Door environment."
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="q_learning",
    choices=["q_learning", "ppo", "dqn", "a2c"],
    help="Model to use for training.",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate for the model.",
)
parser.add_argument(
    "-gamma",
    "--discount_factor",
    type=float,
    default=0.99,
    help="Discount factor for future rewards.",
)
parser.add_argument(
    "-eps",
    "--exploration_rate",
    type=float,
    default=1.0,
    help="Initial exploration rate for epsilon-greedy policy.",
)
parser.add_argument(
    "-opt",
    "--optimistic_init",
    type=float,
    default=0.0,
    help="Optimistic initialization value for Q-network output bias.",
)
parser.add_argument(
    "-eps_decay",
    "--exploration_decay",
    type=float,
    default=0.995,
    help="Decay rate for exploration.",
)
parser.add_argument(
    "-num_ep",
    "--num_episodes",
    type=int,
    default=1000,
    help="Number of training episodes.",
)
parser.add_argument(
    "-bs", "--batch_size", type=int, default=64, help="Batch size for training."
)
parser.add_argument(
    "-tuf",
    "--target_update_frequency",
    type=int,
    default=50,
    help="Frequency of updating the target network (for DQN).",
)
parser.add_argument(
    "-rbs",
    "--replay_buffer_size",
    type=int,
    default=10000,
    help="Size of the replay buffer (for DQN).",
)
parser.add_argument(
    "-burnin",
    "--burnin",
    type=int,
    default=1000,
    help="Number of steps to populate the replay buffer before training (for DQN).",
)
parser.add_argument(
    "-a2c_vlc",
    "--a2c_val_loss_coef",
    type=float,
    default=0.5,
    help="Coefficient for value loss in A2C"
)
parser.add_argument(
    "-a2c_elc",
    "--a2c_entropy_loss_coef",
    type=float,
    default=0.01,
    help="Coefficient for entropy exploration loss in A2C"
)
parser.add_argument(
    "-conv",
    "--convolutional",
    action="store_true",
    help="Use convolutional neural network (for DQN).",
)
parser.add_argument(
    "-test",
    "--test_frequency",
    type=int,
    default=50,
    help="Frequency of testing the model during training.",
)
parser.add_argument(
    "-save",
    "--save_model_frequency",
    type=int,
    default=50,
    help="Frequency of saving the model (weights or table) during training.",
)
parser.add_argument(
    "-viz",
    "--visualisation_frequency",
    type=int,
    default=50,
    help="Frequency of visualising (episode rollouts, value functions etc.) during training.",
)
parser.add_argument(
    "-map",
    "--map_name",
    type=str,
    default="map.txt",
    help="Name of map file in maps folder.",
)
parser.add_argument(
    "-map_yaml",
    "--map_yaml_filename",
    type=str,
    default="map.yaml",
    help="Nme of map YAML file in maps folder.",
)
parser.add_argument(
    "-rep",
    "--representation",
    type=str,
    default="agent_position",
    choices=["agent_position", "pixel"],
    help="State representation to use.",
)
parser.add_argument(
    "-timeout",
    "--episode_timeout",
    type=int,
    default=200,
    help="Episode timeout in steps.",
)
parser.add_argument(
    "-results",
    "--results_dir",
    type=str,
    default="results",
    help="Directory to save results.",
)


def create_experiment_directory(base_dir: str) -> str:
    # Create a unique experiment directory based on timestamp
    experiment_name = datetime.now().strftime("%Y-%d-%m-%H-%M")
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    viz_dir = os.path.join(experiment_dir, "rollouts")
    os.makedirs(viz_dir, exist_ok=True)
    return experiment_dir


def setup_environment(
    map_path: str, map_yaml_path: str, episode_timeout: int, representation: str
):
    env = key_door_env.KeyDoorEnv(
        map_ascii_path=map_path,
        map_yaml_path=map_yaml_path,
        representation=representation,
        episode_timeout=episode_timeout,
    )
    env = visualisation_env.VisualisationEnv(env)

    return env


def setup_model(
    model_type: str,
    state_space,
    action_space,
):
    if model_type == "q_learning":
        return q_learning.QLearning(
            state_space=state_space,
            action_space=action_space,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
        )
    elif model_type == "ppo":
        return ppo.PPO()
    elif model_type == "dqn":
        sample_state = env.reset_environment()
        num_actions = len(action_space)
        return dqn.DQN(
            sample_state=sample_state,
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            batch_size=args.batch_size,
            target_update_frequency=args.target_update_frequency,
            replay_buffer_size=args.replay_buffer_size,
            burnin=args.burnin,
            convolutional=args.convolutional,
            optimistic_init=args.optimistic_init
        )
    elif model_type == "a2c":
        sample_state = env.reset_environment()
        num_actions = len(action_space)
        return a2c.A2C(
            sample_state=sample_state,
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            convolutional=args.convolutional,
            value_loss_coef=args.a2c_val_loss_coef,
            entropy_coef=args.a2c_entropy_loss_coef
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(
    model,
    env,
    num_episodes,
    episode_timeout,
    test_frequency,
    visualisation_frequency,
    save_dir,
):

    episode_lengths = []
    episode_rewards = []
    episode_losses = []

    test_episode_lengths = []
    test_episode_rewards = []

    # Initialize variables for tqdm monitoring
    latest_test_reward = None
    latest_train_loss = None

    pbar = tqdm(range(num_episodes))
    for i in pbar:

        episode_length = 0
        episode_reward = 0
        episode_loss = 0

        state = env.reset_environment()

        for step in range(episode_timeout):
            action = model.select_action(state)
            reward, next_state = env.step(action)

            info = model.step(
                state=state,
                action=action,
                reward=reward,
                new_state=next_state,
                active=env.active,
            )

            state = next_state

            episode_length += 1
            episode_reward += reward
            episode_loss += info.get("loss", np.nan)

            if not env.active:
                break

        if i % test_frequency == 0:
            test_reward, test_episode_length = test(model, env, episode_timeout)
            test_episode_rewards.append(test_reward)
            test_episode_lengths.append(test_episode_length)
            latest_test_reward = test_reward
        if i % visualisation_frequency == 0:
            env.visualise_episode_history(
                save_path=os.path.join(save_dir, "rollouts", f"episode_{i}.mp4"),
                history="test",
            )
        if i % args.save_model_frequency == 0:
            model.save_model(save_dir, i)

        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / episode_length)
        latest_train_loss = episode_loss / episode_length

        # Update tqdm display with latest metrics
        desc_parts = []
        if latest_test_reward is not None:
            desc_parts.append(f"Test Reward: {latest_test_reward:.2f}")
        if latest_train_loss is not None and not np.isnan(latest_train_loss):
            desc_parts.append(f"Train Loss: {latest_train_loss:.6f}")
        if episode_reward is not None:
            desc_parts.append(f"Episode Reward: {episode_reward:.2f}")
        
        if desc_parts:
            pbar.set_description(" | ".join(desc_parts))

    np.savez(
        os.path.join(save_dir, "training_stats.npz"),
        episode_lengths=episode_lengths,
        episode_rewards=episode_rewards,
        test_episode_lengths=test_episode_lengths,
        test_episode_rewards=test_episode_rewards,
        episode_losses=episode_losses,
    )


def test(model, env, episode_timeout):
    state = env.reset_environment(train=False)
    total_reward = 0
    episode_length = 0
    for step in range(episode_timeout):
        action = model.select_greedy_action(state)
        reward, next_state = env.step(action)
        total_reward += reward
        episode_length += 1
        state = next_state
        if not env.active:
            break
    return total_reward, episode_length


if __name__ == "__main__":
    args = parser.parse_args()
    experiment_dir = create_experiment_directory(
        base_dir=os.path.join(current_dir, args.results_dir)
    )

    # save args to experiment_dir
    with open(os.path.join(experiment_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    map_path = os.path.join(current_dir, "maps", args.map_name)
    map_yaml_path = os.path.join(current_dir, "maps", args.map_yaml_filename)
    env = setup_environment(
        map_path=map_path,
        map_yaml_path=map_yaml_path,
        episode_timeout=args.episode_timeout,
        representation=args.representation,
    )
    model = setup_model(
        model_type=args.model,
        state_space=env.state_space,
        action_space=env.action_space,
    )
    train(
        model,
        env,
        args.num_episodes,
        args.episode_timeout,
        args.test_frequency,
        args.visualisation_frequency,
        experiment_dir,
    )
