import os
import random
import math
import numpy as np
from torch.utils.data.sampler import Sampler


# define new sampler in one task
class EpisodeGivenSampler(Sampler):
    def __init__(self, dataset, num_stage, num_stage_step, batch_size, total_iters=None):

        self.dataset = dataset
        self.total_iter = num_stage * num_stage_step
        if total_iters is not None:
            self.total_iter = total_iters
        self.batch_size = batch_size

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices)
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        all_size = self.total_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size


class ImageClass:
    # Stores the paths to images for a given class

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        image_dir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(image_dir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(image_dir):
    image_paths = []
    if os.path.isdir(image_dir):
        images = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir, img) for img in images]
    return image_paths


def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode == 'SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes * (1 - split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode == 'SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class * (1 - split_ratio)))
            if split == nrof_images_in_class:
                split = nrof_images_in_class - 1
            if split >= min_nrof_images_per_class and nrof_images_in_class - split >= 1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)

    return train_set, test_set


def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff
