import numpy as np
import os
import cv2
from dsl.base_dsl import BaseDSL, one_hot_labels
from sklearn.model_selection import train_test_split
from PIL import Image
from dsl.base_dsl import BaseDSL, one_hot_labels
from os.path import expanduser, join
from cfg import cfg


def label_str_to_index(label_str):
    label_map = {
        "bluebell": 0,
        "buttercup": 1,
        "coltsfoot": 2,
        "cowslip": 3,
        "crocus": 4,
        "daffodil": 5,
        "daisy": 6,
        "dandelion": 7,
        "fritillary": 8,
        "iris": 9,
        "lilyvalley": 10,
        "pansy": 11,
        "snowdrop": 12,
        "sunflower": 13,
        "tigerlily": 14,
        "tulip": 15,
        "windflower": 16
    }
    return label_map[label_str]

class Flower17DSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=0.2,
                 normalize_channels=False, path=None, resize=None):

        if mode == 'val':
            assert val_frac is not None

        if path is None:
            self.path = os.path.join(cfg.dataset_dir, 'flower2')
        else:
            self.path = path

        self.resize=(28,28)

        super(Flower17DSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )

    def is_multilabel(self):
        return False



    def load_data(self, mode, val_frac):
        images = []
        labels = []
        if mode=="test":
            mode = "val"
        for filename in os.listdir(os.path.join(self.path, mode)):
            for imgname in os.listdir(os.path.join(self.path, mode, filename)):
                img = cv2.imread(os.path.join(self.path, mode, filename,imgname))
                if self.resize is not None:
                    img = cv2.resize(img, self.resize)
                images.append(img)
                label=label_str_to_index(filename)
                labels.append(label)


        self.data = np.array(images)
        self.labels = np.array(labels)

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)

    def convert_Y(self, Y):
        return one_hot_labels(Y, 17)
