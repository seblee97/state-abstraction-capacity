from typing import List, Tuple, Optional

import numpy as np
from key_door import wrapper


class InvisibleRewardEnv(wrapper.Wrapper):
    """
    Wrapper that suppresses the visual rendering of specified reward positions
    in the pixel observation, while leaving the reward signal intact.

    Reward cells are rendered as red [1,0,0] in the RGB skeleton. This wrapper
    replaces those pixels with white [1,1,1] (free cell colour) before the
    grayscale/transpose/batch pipeline runs, eliminating the visual cue without
    affecting what the agent receives as reward.

    Args:
        env: KeyDoorEnv (before VisualisationEnv wrapping).
        invisible_positions: list of (x, y) positions to suppress visually.
    """

    def __init__(self, env, invisible_positions: List[Tuple[int, int]]):
        super().__init__(env=env)
        # Store as (y, x) for numpy row-major indexing (matching _env_skeleton)
        self._invisible_yx = [tuple(pos[::-1]) for pos in invisible_positions]

    def get_state_representation(self, tuple_state=None):
        state = self._env.get_state_representation(tuple_state=tuple_state)
        if isinstance(state, np.ndarray):
            state = self._blank_reward_pixels(state)
        return state

    def reset_environment(self, train: bool = True, map_yaml_path=None):
        state = self._env.reset_environment(train=train, map_yaml_path=map_yaml_path)
        if isinstance(state, np.ndarray):
            state = self._blank_reward_pixels(state)
        return state

    def step(self, action):
        reward, state = self._env.step(action)
        if isinstance(state, np.ndarray):
            state = self._blank_reward_pixels(state)
        return reward, state

    def _blank_reward_pixels(self, state: np.ndarray) -> np.ndarray:
        """Replace reward pixels with free-cell colour in the processed observation.

        State shape after full pipeline: (1, 1, H, W) — batch x channel x H x W.
        Red [1,0,0] grayscales to 0.299 under BT.601 (0.299R + 0.587G + 0.114B).
        Free cells are white [1,1,1] -> grayscale 1.0.
        """
        RED_GRAY = 0.299
        FREE_GRAY = 1.0

        state = state.copy()
        img = state[0, 0]  # (H, W)
        for (y, x) in self._invisible_yx:
            if abs(img[y, x] - RED_GRAY) < 1e-3:
                img[y, x] = FREE_GRAY
        state[0, 0] = img
        return state
