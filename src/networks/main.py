from .mnist_LeNet import *
from .cifar10_LeNet import *


def build_network(net_name):
    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    elif net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()
    else:
        raise Exception('NetName not found')

    return net


def build_autoencoder(net_name):
    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()
    elif net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()
    else:
        raise Exception('NetName not found')

    return ae_net
