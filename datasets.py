import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100

def load_cifar_10_train(path):
    cifar_10_obj = CIFAR10(path, train = True, download = False)
    x = cifar_10_obj.data
    y = np.array(cifar_10_obj.targets)
    return x, y
#update functions bellow
def load_cifar_10_test():
    cifar_10_path = '/scratch/ssd004/datasets/cifar10'
    cifar_10_obj = CIFAR10(cifar_10_path, train = False, download = False)
    x = cifar_10_obj.data
    y = np.array(cifar_10_obj.targets)
    return x, y

def load_cifar_100_train():
    cifar_100_path = '/scratch/ssd004/datasets/cifar100'
    cifar_100_obj = CIFAR100(cifar_100_path,train = True, download = False)
    x = cifar_100_obj.data
    y = np.array(cifar_100_obj.targets)
    return x, y

def load_cifar_100_test():
    cifar_100_path = '/scratch/ssd004/datasets/cifar100'
    cifar_100_obj = CIFAR100(cifar_100_path,train = False, download = False)
    x = cifar_100_obj.data
    y = np.array(cifar_100_obj.targets)
    return x, y
