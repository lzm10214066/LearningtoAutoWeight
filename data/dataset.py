import torchvision.transforms as transforms
from .cifar10_features import CIFAR10, CIFAR100
import torch.nn.functional as F


# #
class Cifar10:
    def __init__(self, config):
        self.config = config
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(15),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        #
        # #transforms.RandomRotation(15),
        #
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.train_actor_train_dataset = CIFAR10(config=config, root=self.config.data, mode='train_actor-train',
                                                 download=True,
                                                 transform=transform_train)
        self.test_actor_train_dataset = CIFAR10(config=config, root=self.config.data, mode='test_actor-train',
                                                download=True,
                                                transform=transform_train)

        self.train_actor_val_dataset = CIFAR10(config=config, root=self.config.data, mode='train_actor-val',
                                               download=True,
                                               transform=transform_test)

        self.test_actor_test_dataset = CIFAR10(config=config, root=self.config.data, mode='test_actor-test',
                                               download=True,
                                               transform=transform_test)


class Cifar100:
    def __init__(self, config):
        self.config = config
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_actor_train_dataset = CIFAR100(config=config, root=self.config.data, mode='train_actor-train',
                                                  download=True,
                                                  transform=transform_train)
        self.test_actor_train_dataset = CIFAR100(config=config, root=self.config.data, mode='test_actor-train',
                                                 download=True,
                                                 transform=transform_train)

        self.train_actor_val_dataset = CIFAR100(config=config, root=self.config.data, mode='train_actor-val',
                                                download=True,
                                                transform=transform_test)
        self.test_actor_test_dataset = CIFAR100(config=config, root=self.config.data, mode='test_actor-test',
                                                download=True,
                                                transform=transform_test)
