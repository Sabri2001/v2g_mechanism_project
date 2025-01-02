from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    """
    An abstract base class for experiments. Each subclass should implement the `run()` method,
    which executes the core experiment logic and populates `self.results`.
    """

    def __init__(self, config: dict):
        """
        Initializes an experiment with the given configuration.

        Args:
            config (dict): A configuration dictionary holding all necessary parameters for the experiment.
        """
        self.config = config
        self.results = None

    @abstractmethod
    def run(self):
        """
        The main method to be implemented by each experiment subclass.
        It should run the experiment and store results in `self.results`.
        """
        pass
