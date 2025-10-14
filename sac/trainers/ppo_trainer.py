import os
import numpy as np
import torch

from tqdm import tqdm


def update_pbar_desc(pbar, train_loss=None, episode_return=None, test_return=None, episode_length=None,
                     value_pred_std=None, logits_mean_std=None):
    """Update tqdm progress bar description with current metrics."""
    desc_parts = ["Training Progress"]
    
    if train_loss is not None:
        desc_parts.append(f"Loss: {train_loss:.4f}")

    # if value_pred_std is not None:
    #     desc_parts.append(f"Value Std: {value_pred_std:.2e}")

    # if logits_mean_std is not None:
    #     desc_parts.append(f"Logits Mean Std: {logits_mean_std:.2e}")

    if episode_return is not None:
        desc_parts.append(f"Ep Ret: {episode_return:.2f}")

    if test_return is not None:
        desc_parts.append(f"Test Ret: {test_return:.2f}")

    if episode_length is not None:
        desc_parts.append(f"Ep Len: {episode_length}")

    pbar.set_description(" | ".join(desc_parts))


def train(
    model,
    train_env,
    test_env,
    num_steps,
    replay_buffer_size,
    episode_timeout,
    update_epochs,
    test_frequency,
    save_model_frequency,
    visualisation_frequency,
    experiment_dir,
):

    state = train_env.reset_environment()

    episode_returns = []
    episode_lengths = []
    epoch_losses = []

    test_episode_lengths = []
    test_episode_returns = []

    episode_length = 0
    episode_return = 0.0
    global_step = 0
    update_count = 0
    
    # Variables for progress bar logging
    latest_train_loss = None
    latest_episode_return = None
    latest_episode_length = None
    latest_test_return = None
    latest_value_pred_std = None
    latest_logits_mean_std = None

    with tqdm(total=num_steps, desc="Training Progress") as pbar:

        while global_step < num_steps:

            model.reset_buffer()

            # Collect rollout
            for _ in range(replay_buffer_size):
                with torch.no_grad():
                    action, logp, _, value = model.select_action(state)
                action = action.item()

                logp   = float(logp.item())
                value  = float(value.item())

                reward, next_state = train_env.step(action)

                model.add_to_buffer(
                    state=state,
                    action=action,
                    reward=reward,
                    active=train_env.active,
                    logp=logp,
                    value=value,
                )

                episode_return += reward
                episode_length += 1

                state = next_state
                global_step += 1
                pbar.update(1)

                if not train_env.active:
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    latest_episode_return = episode_return
                    latest_episode_length = episode_length
                    # Update progress bar description with current metrics
                    update_pbar_desc(pbar, latest_train_loss, latest_episode_return, latest_test_return, latest_episode_length,
                                     value_pred_std=latest_value_pred_std, logits_mean_std=latest_logits_mean_std)
                    episode_return = 0.0
                    episode_length = 0
                    state = train_env.reset_environment()

                if global_step >= num_steps:
                    break

            # Bootstrap value for last state (should be 0 if episode ended)
            with torch.no_grad():
                if train_env.active:
                    _, last_value = model.forward(state)
                    last_value = float(last_value.item())
                else:
                    last_value = 0.0

            model.compute_gae(last_value)

            # PPO Update
            for _ in range(update_epochs):
                info = model.step()
                latest_train_loss = info.get("loss", np.nan)
                epoch_losses.append(latest_train_loss)
                latest_value_pred_std = info.get("values_std", np.nan)
                latest_logits_mean_std = info.get("logits_mean_std", np.nan)
                # Update progress bar description with current metrics
                update_pbar_desc(pbar, latest_train_loss, latest_episode_return, latest_test_return, latest_episode_length,
                                 value_pred_std=latest_value_pred_std, logits_mean_std=latest_logits_mean_std)

            update_count += 1

            # Periodic testing and saving
            if update_count % test_frequency == 0 or global_step >= num_steps:
                test_return, test_episode_length = test(model, test_env, episode_timeout)
                test_episode_returns.append(test_return)
                test_episode_lengths.append(test_episode_length)
                latest_test_return = test_return

            # Periodic visualization
            if update_count % visualisation_frequency == 0 or global_step >= num_steps:
                # Ensure rollouts directory exists
                rollouts_dir = os.path.join(experiment_dir, "rollouts")
                os.makedirs(rollouts_dir, exist_ok=True)

                train_env.visualise_episode_history(
                    save_path=os.path.join(rollouts_dir, f"train_episode_{global_step}.mp4"),
                    history="train",
                )

                test_env.visualise_episode_history(
                    save_path=os.path.join(rollouts_dir, f"test_episode_{global_step}.mp4"),
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
