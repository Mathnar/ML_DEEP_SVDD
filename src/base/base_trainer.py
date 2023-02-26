from abc import ABC, abstractmethod
from .base_dataset import BaseADDataset
from .base_net import BaseNet


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        """
        Implement train method that trains the given network using the training_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        """
        Implement test method that evaluates the testing_set of dataset on the given network.
        """
        pass
