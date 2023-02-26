from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_data


def load_dataset(dataset_name, data_path, normal_class):
    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    elif dataset_name == 'cifar10':
        dataset = CIFAR10_data(root=data_path, normal_class=normal_class)
    else:
        raise Exception("Dataset not found")
    return dataset
