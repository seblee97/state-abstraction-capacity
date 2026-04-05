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

    def _blank_reward_pixels(self, state: np.ndarray) -> np.ndarray:
        """Zero out reward pixels in the pixel observation.

        The pixel pipeline is: RGB (H x W x 3) -> grayscale (H x W x 1)
        -> transpose (1 x H x W) -> batch (1 x 1 x H x W).
        By the time we see the state it is already fully processed.

        We identify reward pixels by their grayscale value: red [1,0,0]
        grayscales to 0.2126 (standard luminance). We replace those pixels
        with the free-cell grayscale value of 1.0.
        """
        state = state.copy()

        # State shape after full pipeline: (batch, C, H, W) = (1, 1, H, W)
        # Remove batch dim to work with (1, H, W), then restore
        img = state[0]  # (1, H, W)

        RED_GRAY = 0.2126  # grayscale of [1, 0, 0] under standard luminance
        FREE_GRAY = 1.0    # grayscale of [1, 1, 1]

        for (y, x) in self._invisible_yx:
            if img[0, y, x] == RED_GRAY:
                img[0, y, x] = FREE_GRAY

        state[0] = img
        return state
