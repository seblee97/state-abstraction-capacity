import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# from sac.models import base


# class PPO(base.BaseModel):

#     def __init__(
#         self,
#         learning_rate: float,
#         discount_factor: float,
#         gae_lambda: float,
#         clip_coef: float,
#     ):
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.gae_lambda = gae_lambda
#         self.clip_coef = clip_coef
#         super().__init__()

#     def select_action(self, state):
#         pass

#     def step(self):
#         pass

#     def save_model(self, path: str) -> None:
#         pass


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
    episode_length = 0
    episode_return = 0.0
    global_step = 0

    while global_step < num_steps:

        model.reset_buffer()

        # Collect rollout
        for _ in range(replay_buffer_size):
            with torch.no_grad():
                action, logp, _, value = model.select_action(state)
            action = action.item()
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

            if not env.active or episode_length >= episode_timeout:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                episode_return = 0.0
                episode_length = 0
                state = env.reset_environment()

            if global_step >= num_steps:
                break

        # Bootstrap value for last state
        with torch.no_grad():
            _, last_value = model.forward(state)

        model.compute_gae(last_value)

        # PPO Update
        for _ in range(update_epochs):
            model.step()

        # if len(ep_returns) > 0 and (global_step // rollout_steps) % 5 == 0:
        #     avg = np.mean(ep_returns[-10:])
        #     fps = int((global_step) / (time.time() - start_time + 1e-8))
        #     print(f"step={global_step} | avg_return_last10={avg:.1f} | fps={fps}")


# def ppo(
#     update_epochs=10,
#     vf_coef=0.5,
#     ent_coef=0.01,
#     max_grad_norm=0.5,
# ):
#     obs_space = env.observation_space
#     act_space = env.action_space
#     assert len(obs_space.shape) == 1, "This basic script expects 1D observations."
#     assert hasattr(act_space, "n"), "This basic script is for discrete action spaces."

#     obs_dim = obs_space.shape[0]
#     act_dim = act_space.n

#     device = torch.device(device)
#     net = ActorCritic(obs_dim, act_dim).to(device)
#     opt = optim.Adam(net.parameters(), lr=lr, eps=1e-5)

#     buffer = RolloutBuffer(rollout_steps, obs_dim, device)

#     obs, _ = env.reset(seed=seed)
#     obs = torch.tensor(obs, dtype=torch.float32, device=device)
#     ep_returns = []
#     ep_len = 0
#     ep_ret = 0.0
#     global_step = 0

#     start_time = time.time()

#     env.close()
#     # quick evaluation
#     eval_env = gym.make(env_id)
#     eval_obs, _ = eval_env.reset(seed=seed + 1)
#     eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device=device)
#     returns = []
#     for _ in range(10):
#         done = False
#         ep_r = 0.0
#         while not done:
#             with torch.no_grad():
#                 act, _v = net.evaluate(eval_obs.unsqueeze(0))
#             next_obs, r, terminated, truncated, _ = eval_env.step(int(act.item()))
#             done = terminated or truncated
#             ep_r += r
#             eval_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
#             if done:
#                 returns.append(ep_r)
#                 eval_obs, _ = eval_env.reset()
#                 eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device=device)
#     eval_env.close()
#     print(
#         f"Evaluation mean return over 10 episodes: {np.mean(returns):.1f} ± {np.std(returns):.1f}"
#     )


# if __name__ == "__main__":
#     # Example: python ppo_basic.py
#     ppo(
#         env_id="CartPole-v1",
#         total_steps=100_000,
#         rollout_steps=2048,
#         update_epochs=10,
#         minibatch_size=64,
#         device="cpu",
#     )
