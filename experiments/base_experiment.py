from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    def __init__(self, config):
        self.config = config
        self.results = None

    @abstractmethod
    def run(self):
        pass
