import numpy as np
import os
import struct
from PIL import Image
from dsl.base_dsl import BaseDSL, one_hot_labels
from os.path import expanduser, join
from cfg import cfg
import numpy as np, os, struct
import matplotlib.pyplot as plt
import csv
import matplotlib
import skimage.data
import skimage.transform


class GTSRDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=0.2,
                 normalize_channels=False, path=None, resize=None):
        self.resize=(28,28)
        if mode == 'val':
            assert val_frac is not None

        if path is None:

            self.path = os.path.join(cfg.dataset_dir, 'GTSRB')
        else:
            self.path = path

        super(GTSRDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )

    # The German Traffic Sign Recognition Benchmark
    #
    # sample code for reading the traffic sign images and the
    # corresponding labels
    #
    # example:
    #
    # trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
    # print len(trainLabels), len(trainImages)
    # plt.imshow(trainImages[42])
    # plt.show()
    #
    # have fun, Christian

    # function for reading the images
    # arguments: path to the traffic sign data, for example './GTSRB/Training'
    # returns: list of images, list of corresponding labels
    # def readTrafficSigns(rootpath):
    #     '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    #
    #     Arguments: path to the traffic sign data, for example './GTSRB/Training'
    #     Returns:   list of images, list of corresponding labels'''
    #     images = []  # images
    #     labels = []  # corresponding labels
    #     # loop over all 42 classes
    #     for c in range(0, 2):
    #         prefix = rootpath + '/' + format(c, '05d') + '/'  # 先打开类文件夹
    #         gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # 再打开csv注释文件
    #         gtReader = csv.reader(gtFile, delimiter=';')  # 用分号隔开每个元素，下一步可以按行读取
    #         next(gtReader)  # 跳过第一行
    #         # 读取csv文件中的每一行并且提取出来第一个元素是要打开的文件名，第八行是标签
    #         for row in gtReader:
    #             images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
    #             labels.append(row[7])  # the 8th column is the label
    #         gtFile.close()
    #     images = [skimage.transform.resize(image, (28, 28), mode='constant')
    #               for image in images]
    #     print(type(images[0]), type(labels[0]))
    #     return images, labels



    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        if mode == 'test':
            folder_name = os.path.join(self.path, 'Final_Test', 'Images')
            label_file = os.path.join(self.path, 'GT-final_test.csv')
            labels=[]
            data = []
            for root, dirs, files in os.walk(folder_name):
                for file in files:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path).convert('RGB')
                    if self.resize is not None:
                        img = img.resize(self.resize, Image.ANTIALIAS)
                    data.append(np.asarray(img))
            with open(label_file, 'r') as file:
                next(file)  # Skip the header row
                for line in file:
                    _, _, _, _, _, _, _, label = line.strip().split(';')
                    labels.append(int(label))


        else:
            assert mode == 'train' or mode == 'val'
            folder_name = os.path.join(self.path, 'Final_Training', 'Images')
            # z = self.readTrafficSigns(folder_name)
            images = []  # images
            labels = []  # corresponding labels
            # loop over all 42 classes
            for c in range(0, 43):
                prefix = folder_name + '/' + format(c, '05d') + '/'
                gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
                gtReader = csv.reader(gtFile, delimiter=';')
                next(gtReader)

                for row in gtReader:
                    images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
                    labels.append(int(row[7]))  # the 8th column is the label
                gtFile.close()
            images = [skimage.transform.resize(image, (28, 28), mode='constant')
                      for image in images]
            print(type(images[0]), type(labels[0]))
            data, labels = images,labels

        self.data = np.array(data)
        self.labels = np.array(labels)

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)


    def convert_Y(self, Y):
        return one_hot_labels(Y, 43)
