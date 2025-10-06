from sac.models import q_learning, ppo, dqn, a2c
from sac.trainers import episodic_trainer, ppo_trainer
from key_door import key_door_env, visualisation_env
import argparse
import numpy as np
import os
from datetime import datetime


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
    "-ns",
    "--num_steps",
    type=int,
    default=100000,
    help="Number of training steps.",
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
    help="Size of the replay buffer (for DQN). Also acts as rollout size for PPO.",
)
parser.add_argument(
    "-burnin",
    "--burnin",
    type=int,
    default=1000,
    help="Number of steps to populate the replay buffer before training (for DQN).",
)
parser.add_argument(
    "-gae",
    "--gae_lambda",
    type=float,
    default=0.95,
    help="GAE lambda parameter (for PPO).",
)
parser.add_argument(
    "-ue",
    "--update_epochs",
    type=int,
    default=10,
    help="Number of epochs to update the policy (for PPO).",
)
parser.add_argument(
    "-vfc",
    "--value_function_coef",
    type=float,
    default=0.5,
    help="Coefficient for value function loss (for PPO).",
)
parser.add_argument(
    "-ec",
    "--entropy_coef",
    type=float,
    default=0.01,
    help="Coefficient for entropy bonus (for PPO).",
)
parser.add_argument(
    "-cc",
    "--clip_coef",
    type=float,
    default=0.2,
    help="Clipping coefficient for PPO.",
)
parser.add_argument(
    "-mgn",
    "--max_grad_norm",
    type=float,
    default=0.5,
    help="Maximum gradient norm for clipping (for PPO).",
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
        sample_state = env.reset_environment()
        num_actions = len(action_space)
        return ppo.PPO(
            sample_state=sample_state,
            num_actions=num_actions,
            batch_size=args.batch_size,
            replay_buffer_size=args.replay_buffer_size,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            vf_coef=args.value_function_coef,
            ent_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            convolutional=args.convolutional,
        )
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

    if args.model in ["q_learning", "a2c", "dqn"]:
        episodic_trainer.train(
            model=model,
            env=env,
            num_episodes=args.num_episodes,
            episode_timeout=args.episode_timeout,
            test_frequency=args.test_frequency,
            save_model_frequency=args.save_model_frequency,
            visualisation_frequency=args.visualisation_frequency,
            experiment_dir=experiment_dir,
        )
    elif args.model == "ppo":
        ppo_trainer.train(
            model=model,
            env=env,
            num_steps=args.num_steps,
            episode_timeout=args.episode_timeout,
            replay_buffer_size=args.replay_buffer_size,
            update_epochs=args.update_epochs,
            test_frequency=args.test_frequency,
            save_model_frequency=args.save_model_frequency,
            visualisation_frequency=args.visualisation_frequency,
            experiment_dir=experiment_dir,
        )
