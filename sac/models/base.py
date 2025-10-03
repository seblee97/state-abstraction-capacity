import abc


class BaseModel(abc.ABC):

    @abc.abstractmethod
    def select_action(self, state, explore=True):
        pass

    @abc.abstractmethod
    def step(self):
        pass