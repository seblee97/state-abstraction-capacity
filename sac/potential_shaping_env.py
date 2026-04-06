from collections import deque
from typing import Tuple

import numpy as np
from key_door import wrapper


class PotentialShapingEnv(wrapper.Wrapper):
    """
    Potential-based reward shaping wrapper.

    Adds F(s, s') = gamma * phi(s') - phi(s) to every environment reward,
    where phi(s) = -shaping_scale * BFS_distance(s, goal).

    This is guaranteed to preserve the optimal policy (Ng et al., 1999).
    Moving closer to the goal gives a positive bonus; moving away gives a
    negative bonus. The shaping signal sums to zero over any complete episode
    that starts and ends at the same state, so it cannot create spurious loops.

    Args:
        env: wrapped key_door environment (should already be wrapped by
             VisualisationEnv if visualisation is needed).
        goal: (x, y) goal position to compute distances from.
        shaping_scale: multiplier on the potential. Higher = stronger shaping.
        gamma: discount factor (must match the agent's gamma).
        map_ascii_path: path to the ASCII map for BFS.
    """

    def __init__(
        self,
        env,
        goal: Tuple[int, int],
        shaping_scale: float,
        gamma: float,
        map_ascii_path: str,
    ):
        super().__init__(env=env)
        self._goal = goal
        self._shaping_scale = shaping_scale
        self._gamma = gamma
        self._distances = self._bfs_distances(map_ascii_path, goal)
        self._prev_position = None

    def _bfs_distances(self, map_ascii_path: str, goal: Tuple[int, int]):
        with open(map_ascii_path) as f:
            grid = [line.rstrip('\n') for line in f]
        H = len(grid)

        def is_free(x, y):
            return 0 <= y < H and 0 <= x < len(grid[y]) and grid[y][x] != '#'

        dist = {}
        queue = deque([(goal, 0)])
        dist[goal] = 0
        while queue:
            (x, y), d = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in dist and is_free(nx, ny):
                    dist[(nx, ny)] = d + 1
                    queue.append(((nx, ny), d + 1))
        return dist

    def _phi(self, position: Tuple[int, int]) -> float:
        d = self._distances.get(tuple(position), None)
        if d is None:
            return 0.0
        return -self._shaping_scale * d

    def reset_environment(self, train: bool = True, map_yaml_path=None):
        state = self._env.reset_environment(train=train, map_yaml_path=map_yaml_path)
        self._prev_position = tuple(self._env.agent_position)
        return state

    def step(self, action) -> Tuple[float, tuple]:
        reward, next_state = self._env.step(action)
        next_position = tuple(self._env.agent_position)
        shaping = self._gamma * self._phi(next_position) - self._phi(self._prev_position)
        self._prev_position = next_position
        return reward + shaping, next_state
