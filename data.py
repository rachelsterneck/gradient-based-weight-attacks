import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


def load_data(dataset_name, dataset_path, batch_size, num_workers, device = torch.device):
    """
    Loads a dataset based on its name.

    :param dataset_name: CIFAR10, CIFAR100, or MNIST
    :param batch_size: The batch size
    :param device: The device to put tensors on
    :return: DataLoaders for test and train data
    """

    if dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        testset = CIFAR10(root=dataset_path, train=False,
                                            download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "CIFAR100":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        testset = CIFAR100(root=dataset_path, train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "MNIST":
        transform = transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            
        testset = torchvision.datasets.MNIST(root=dataset_path, train=False,
                                            download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])  
        testset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform)

        num_classes = 1000
    else:
        raise ValueError("{} is not a valid dataset.".format(dataset_name))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    return testloader, num_classes
