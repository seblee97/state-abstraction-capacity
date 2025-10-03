from sac.models import base 

import numpy as np

class QLearning(base.BaseModel):
    
    def __init__(
            self, 
            action_space, 
            state_space, 
            learning_rate: float, 
            discount_factor: float, 
            exploration_rate: float, 
            exploration_decay: float
        ):

        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}
        self._id_state_mapping = {i: state for i, state in enumerate(self._state_space)}

        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay

        # zero-initialize for now
        self._state_action_values = np.zeros((len(self._state_space), len(self._action_space)))

    def select_action(self, state):
        if np.random.rand() < self._exploration_rate:
            action = np.random.choice(self._action_space)
        else:
            action = self.select_greedy_action(state)
        return action
    
    def select_greedy_action(self, state):
        state_id = self._state_id_mapping[state]
        action = np.argmax(self._state_action_values[state_id])
        return action

    def step(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        new_state: tuple[int, int],
        active: bool,
    ):

        state_id = self._state_id_mapping[state]
        new_state_id = self._state_id_mapping[new_state]

        if active:
            discount = self._discount_factor
        else:
            discount = 0

        initial_value = self._state_action_values[state_id][action]
        new_sate_values = self._state_action_values[new_state_id]
        
        td_error = reward + discount * np.max(new_sate_values) - initial_value

        updated_value = initial_value + self._learning_rate * td_error
        self._state_action_values[state_id][action] = updated_value

        return {"loss": td_error ** 2}


    def save_model(self, path):
        # Save the Q-table to a file
        pass