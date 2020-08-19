from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
           (1 - mixing_ratio) * np.eye(num_classes)



# from https://github.com/xjtushujun/meta-weight-net
def get_img_num_per_cls(dataset, imb_factor=None, num=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = num / 10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = num / 100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, config, root, mode='train_actor-val',
                 transform=None, target_transform=None,
                 download=False):
        self.config = config
        avaliable_list = ['train_actor-train', 'train_actor-val', 'test_actor-train', 'test_actor-test']
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode  # training set or test set

        #
        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        if self.mode not in avaliable_list:
            print('error mode value')
            exit(-1)
        if self.mode == 'test_actor-test':
            downloaded_list = self.test_list
        else:
            downloaded_list = self.train_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        num_classes = 10
        if self.config.dataset == 'cifar100':
            num_classes = 100

        if 'test' not in self.mode:
            data_s = [[] for _ in range(num_classes)]
            target_s = [[] for _ in range(num_classes)]

            num_data = len(self.targets)
            for i in range(num_data):
                data_s[self.targets[i]].append(self.data[i])
                target_s[self.targets[i]].append(self.targets[i])

            self.data = []
            self.targets = []

        if self.mode == 'train_actor-val':
            num_val = int(self.config.val_num / num_classes)
            for k in range(num_classes):
                data_s[k] = data_s[k][-num_val:]
                target_s[k] = target_s[k][-num_val:]
                self.data.extend(data_s[k])
                self.targets.extend(target_s[k])

            self.data = np.array(self.data)
            self.targets = np.array(self.targets).flatten()

        if self.mode == 'train_actor-train':
            num_train = int(self.config.train_num / num_classes)
            for k in range(num_classes):
                data_s[k] = data_s[k][0:num_train]
                target_s[k] = target_s[k][0:num_train]
                self.data.extend(data_s[k])
                self.targets.extend(target_s[k])

            self.data = np.array(self.data)
            self.targets = np.array(self.targets).flatten()

        if self.mode == 'train_actor-train' or self.mode == "test_actor-train":
            if self.config.label_noise > 0:
                C = uniform_mix_C(self.config.label_noise, num_classes)
                print(C)
                self.C = C

                for i in range(len(self.targets)):
                    self.targets[i] = np.random.choice(num_classes, p=C[self.targets[i]])

                # high = 9
                # if self.config.dataset == 'cifar100':
                #     high = 99
                # for i in range(len(self.targets)):
                #     e = np.random.random_sample()
                #     if e <= self.config.label_noise:
                #         label_old = self.targets[i]
                #         label_new = np.random.random_integers(0, high)
                #         while label_new == label_old:
                #             label_new = np.random.random_integers(0, high)
                #         self.targets[i] = label_new

            if self.config.imbalance:
                num_data = len(self.targets)
                img_num_list = get_img_num_per_cls(self.config.dataset, self.config.imb_factor, num_data)
                print(self.mode, ":", img_num_list)

                data_s = [[] for _ in range(num_classes)]
                target_s = [[] for _ in range(num_classes)]

                for i in range(num_data):
                    data_s[self.targets[i]].append(self.data[i])
                    target_s[self.targets[i]].append(self.targets[i])

                self.data = []
                self.targets = []

                for k in range(num_classes):
                    data_s[k] = data_s[k][0:img_num_list[k]]
                    target_s[k] = target_s[k][0:img_num_list[k]]
                    self.data.extend(data_s[k])
                    self.targets.extend(target_s[k])

                self.data = np.array(self.data)
                self.targets = np.array(self.targets).flatten()
                print(self.mode, ' imbalance shape:', self.data.shape)
                print(self.mode, ' imbalance shape:', self.targets.shape)

        self._load_meta()

        # if self.mode == 'train' and self.config.get_feature_from_dumped_file:
        #     # load feature file
        #     self.features_file_path = os.path.join(self.root, 'cifar10_features_train.pkl')
        #     with open(self.features_file_path, 'rb') as f:
        #         if sys.version_info[0] == 2:
        #             entry = pickle.load(f)
        #         else:
        #             entry = pickle.load(f, encoding='latin1')
        #         self.features = entry['features']
        #         self.features = self.features[0:self.data.shape[0]]
        #
        #     assert self.data.shape[0] == self.features.shape[0]

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.mode == 'train' and self.config.get_feature_from_dumped_file:
            feature = self.features[index]
            return img, target, feature
        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
