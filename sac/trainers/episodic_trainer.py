import os
import numpy as np

from tqdm import tqdm

from sac import utils


def train(
    model,
    train_env,
    test_env,
    num_episodes,
    episode_timeout,
    test_frequency,
    save_model_frequency,
    visualisation_frequency,
    experiment_dir,
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

        state_ = train_env.reset_environment()
        state = state_[:2]

        for step in range(episode_timeout):
            action = model.select_action(state)
            reward, next_state_ = train_env.step(action)
            next_state = next_state_[:2]

            info = model.step(
                state=state,
                action=action,
                reward=reward,
                new_state=next_state,
                active=train_env.active,
            )

            state = next_state

            episode_length += 1
            episode_reward += reward
            episode_loss += info.get("loss", np.nan)

            if not train_env.active:
                break

        if i % test_frequency == 0:
            test_reward, test_episode_length = test(model, test_env, episode_timeout)
            test_episode_rewards.append(test_reward)
            test_episode_lengths.append(test_episode_length)
            latest_test_reward = test_reward
        if i % visualisation_frequency == 0:
            train_env.visualise_episode_history(
                save_path=os.path.join(
                    experiment_dir, "rollouts", f"train_episode_{i}.mp4"
                ),
                history="train",
            )
            test_env.visualise_episode_history(
                save_path=os.path.join(
                    experiment_dir, "rollouts", f"test_episode_{i}.mp4"
                ),
                history="test",
            )
        if i % save_model_frequency == 0:
            model.save_model(experiment_dir, i)

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
        os.path.join(experiment_dir, "training_stats.npz"),
        episode_lengths=episode_lengths,
        episode_rewards=episode_rewards,
        test_episode_lengths=test_episode_lengths,
        test_episode_rewards=test_episode_rewards,
        episode_losses=episode_losses,
    )


def test(model, env, episode_timeout):
    state_ = env.reset_environment(train=False)
    state = state_[:2]
    total_reward = 0
    episode_length = 0
    for step in range(episode_timeout):
        action = model.select_greedy_action(state)
        reward, next_state_ = env.step(action)
        next_state = next_state_[:2]
        total_reward += reward
        episode_length += 1
        state = next_state
        if not env.active:
            break
    return total_reward, episode_length
