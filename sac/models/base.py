import abc


class BaseModel(abc.ABC):

    @abc.abstractmethod
    def select_action(self, state):
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save_model(self, path: str, episode: int):
        pass
