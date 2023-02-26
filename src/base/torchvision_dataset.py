from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader):
        train_loader = DataLoader(dataset=self.training_set, batch_size=batch_size, shuffle=shuffle_train)
        test_loader = DataLoader(dataset=self.testing_set, batch_size=batch_size, shuffle=shuffle_test)
        return train_loader, test_loader
