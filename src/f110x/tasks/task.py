from abc import abstractmethod, ABC


# TODO: align Task interface with experiment registry (scenario-driven instantiation, logging hooks).
class Task(ABC):

    @abstractmethod
    def reward(self, agent_id, state, action) -> float:
        pass

    @abstractmethod
    def done(self, agent_id, state) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    def act(self, agent_id, observation, action_space):
        raise NotImplementedError(f"Task {self.__class__.__name__} does not implement act()")
