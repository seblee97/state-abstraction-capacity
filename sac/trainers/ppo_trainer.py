import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def train(
    model,
    env,
    num_steps,
    replay_buffer_size,
    episode_timeout,
    update_epochs,
    test_frequency,
    save_model_frequency,
    visualisation_frequency,
    experiment_dir,
):

    state = env.reset_environment()

    episode_returns = []
    episode_lengths = []
    epoch_losses = []

    test_episode_lengths = []
    test_episode_returns = []

    episode_length = 0
    episode_return = 0.0
    global_step = 0
    update_count = 0

    while global_step < num_steps:

        model.reset_buffer()

        # Collect rollout
        for _ in range(replay_buffer_size):
            with torch.no_grad():
                action, logp, _, value = model.select_action(state)
            action = action.item()

            # if not env.active:
            #     import pdb

            #     pdb.set_trace()
            reward, next_state = env.step(action)

            model.add_to_buffer(
                state=state,
                action=action,
                logp=logp,
                reward=reward,
                active=env.active,
                value=value,
            )

            episode_return += reward
            episode_length += 1

            state = next_state
            global_step += 1

            if not env.active:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                episode_return = 0.0
                episode_length = 0
                state = env.reset_environment()

            if global_step >= num_steps:
                break

        # Bootstrap value for last state (should be 0 if episode ended)
        with torch.no_grad():
            if env.active:
                _, last_value = model.forward(state)
            else:
                last_value = torch.tensor(0.0).to(next(model._net.parameters()).device)

        model.compute_gae(last_value)

        # PPO Update
        for _ in range(update_epochs):
            info = model.step()
            epoch_losses.append(info.get("loss", np.nan))

        update_count += 1

        # Periodic testing and saving
        if update_count % test_frequency == 0 or global_step >= num_steps:
            test_return, test_episode_length = test(model, env, episode_timeout)
            test_episode_returns.append(test_return)
            test_episode_lengths.append(test_episode_length)

        # Periodic visualization
        if update_count % visualisation_frequency == 0 or global_step >= num_steps:
            # Ensure rollouts directory exists
            rollouts_dir = os.path.join(experiment_dir, "rollouts")
            os.makedirs(rollouts_dir, exist_ok=True)

            env.visualise_episode_history(
                save_path=os.path.join(rollouts_dir, f"episode_{global_step}.mp4"),
                history="test",
            )

        # Periodic model saving
        if update_count % save_model_frequency == 0 or global_step >= num_steps:
            model.save_model(experiment_dir, global_step)

    np.savez(
        os.path.join(experiment_dir, "training_stats.npz"),
        episode_lengths=episode_lengths,
        episode_returns=episode_returns,
        test_episode_lengths=test_episode_lengths,
        test_episode_returns=test_episode_returns,
        epoch_losses=epoch_losses,
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
