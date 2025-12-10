import os
from sac.models import base
from sac import utils

import numpy as np


class QuotientQLearning(base.BaseModel):

    def __init__(
        self,
        state_space,
        state_blocks,
        state_label,
        sa_label,
        action_index_per_block,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        exploration_decay: float,
    ):

        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}

        self._state_label = state_label
        self._sa_label = sa_label
        self._action_index_per_block = action_index_per_block

        self._Q_tilde, self._valid_mask, self._to_abstract, self._to_base_action = (
            utils.build_quotient_Q_struct(
                state_blocks, state_label, sa_label, action_index_per_block
            )
        )

        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay

        super().__init__()

    def select_action(self, state):
        state_id = self._state_id_mapping[state]
        b = self._state_label[state_id]
        if np.random.rand() < self._exploration_rate:
            valid_is = np.flatnonzero(self._valid_mask[b])
            action = self._to_base_action(int(np.random.choice(valid_is)), state_id)
        else:
            action = self.select_greedy_action(state)
        return action

    def select_greedy_action(self, state):
        state_id = self._state_id_mapping[state]
        b = self._state_label[state_id]
        q_row = np.where(self._valid_mask[b], self._Q_tilde[b], -np.inf)
        i = int(np.argmax(q_row))
        return self._to_base_action(i, state_id)

    def step(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        new_state: tuple[int, int],
        active: bool,
    ):
        # (s,a) -> (b,i)
        state_id = self._state_id_mapping[state]
        new_state_id = self._state_id_mapping[new_state]

        b = self._state_label[state_id]
        sa_cls = self._sa_label[state_id, action]
        i = self._action_index_per_block[b][sa_cls]

        if active:
            discount = self._discount_factor
        else:
            discount = 0

        # target uses next abstract state b'
        b_next = self._state_label[new_state_id]
        q_next = np.where(self._valid_mask[b_next], self._Q_tilde[b_next], -np.inf)
        td_target = reward + discount * np.max(q_next)
        td_error = td_target - self._Q_tilde[b, i]

        self._Q_tilde[b, i] += self._learning_rate * td_error

        return {"loss": td_error**2}

    def save_model(self, path: str, episode: int) -> None:
        # Save the Q-table to a file
        np.savez(os.path.join(path, f"q_table_{episode}.npz"), self._Q_tilde)
