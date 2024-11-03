import abc


class EnvBase(abc.ABC):
    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def get_dims_dict(self):
        pass

    @abc.abstractmethod
    def get_action_range(self):
        pass
