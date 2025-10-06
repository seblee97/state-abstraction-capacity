import os
import numpy as np


def train(
    model,
    env,
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

    for i in range(num_episodes):

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
        if i % visualisation_frequency == 0:
            env.visualise_episode_history(
                save_path=os.path.join(experiment_dir, "rollouts", f"episode_{i}.mp4"),
                history="test",
            )
        if i % save_model_frequency == 0:
            model.save_model(experiment_dir, i)

        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / episode_length)

    np.savez(
        os.path.join(experiment_dir, "training_stats.npz"),
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
