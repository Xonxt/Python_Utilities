##
# ingroup: ML_Utilities
# file:    data_generator.py
# brief:   This file contains the class for image generation for Keras training
# author:  Nikita Kovalenko (mykyta.kovalenko@hhi.fraunhofer.de)
# date:    02.09.2020
#
# Copyright:
# 2020 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
# The copyright of this software source code is the property of HHI.
# This software may be used and/or copied only with the written permission
# of HHI and in accordance with the terms and conditions stipulated
# in the agreement/contract under which the software has been supplied.
# The software distributed under this license is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
##

#----------------------
import numpy as np
import cv2
import copy
import math
import os
import time, datetime
import traceback
#----------------------
from utils.utils import *
from keras_stuff.augmentations import *
#----------------------
from keras.utils import (Sequence, to_categorical)

#----------------------

class DataGenerator(Sequence):
    def __init__(self, data_x, labels, one_hot_encode=True,
                 batch_size=16, shuffle=True, input_size=(368,368), padding=False,
                 augment=True, probability=0.5, preprocessing=None, dataset_size=-1):
        """
        Initialize the Image Data Generator

        Arguments:
        data_x          The list or Numpy Array of input images
        labels          The list of output labels
        one_hot_encode  Use 'one-hot' encoding on the labels ("0 0 1, 0 1 0 and 1 0 0" instead of "0, 1 and 2")
        shuffle         Shuffle the data on every epoch
        input_size      Resize the images to this input size
        padding         Use padding for the resized images (pad to a square shape)
        augment         Use random augmentations for the images
        probability     Default probability for each augmentation
        preprocessing   Use some sort of preprocessing for the images
        """

        self.data_x = data_x
        self.labels = labels

        self.one_hot_encode = one_hot_encode

        # one-hot encode the labels, if necessary:
        if self.one_hot_encode:
            self.labels = to_categorical(self.labels)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.padding = padding
        self.augment = augment
        self.probability = probability
        self.preprocessing = preprocessing

        # generate the index list, and shuffle the data if necessary
        self.on_epoch_end()

    def __len__(self):
        """
        Get the number of training steps (dataset size, divided by batch size)
        """
        return len(self.data_x) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        # Generate indices of the batch
        batch_indices = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        images, labels = self.__get_data(batch_indices)

        return images, labels

    def __get_data(self, batch_indices):

        BATCH_IMAGES = []
        BATCH_LABELS = []

        for index in batch_indices:
            # get an image, and the label:
            image = copy.deepcopy(self.data_x[index])
            label = copy.deepcopy(self.labels[index])

            # perform augmentations, if necessary. Comment out the functions, that you don't want
            if self.augment:
                # First, the color-based augmentations:
                augment_add_noise(image, max_chance=self.probability, max_intensity=0.1)
                augment_add_resize_artifacts(image, max_chance=self.probability, max_factor=4)
                augment_contrast(image, max_chance=self.probability, max_range=0.2)
                augment_brightness(image, max_chance=self.probability, max_range=30)
                augment_gamma(image, max_chance=self.probability, max_range=(0.75, 2))
                augment_colorize(image, max_chance=self.probability, max_rate=0.1)

            # then, scale the image to the desired size AFTER the color augmentations:
            image, _ = resize_ratio(image, shape=self.input_size, padding=self.padding, pad_value=128)

            if self.augment:
                # and now, do the  Affine transformations
                # (so that color-transformations don't affect the padding value)
                augment_flip(image, max_chance=self.probability, vertical=True)
                augment_translate(image, max_chance=self.probability, max_range=[-0.1, 0.2])
                augment_shear(image, max_chance=self.probability, max_range=0.2, horizontal=True, vertical=True)
                augment_rotate(image, max_chance=self.probability, max_angle=45)
                augment_scale(image, max_chance=self.probability, max_range=0.2)

            # if provided some preprocessing function, use it on the image:
            if self.preprocessing is not None and callable(self.preprocessing):
                image = self.preprocessing(image)

            # finally, add to the list:
            BATCH_IMAGES.append(image)
            BATCH_LABELS.append(label)

        # and finally, return the image and label tensor:
        IMAGES_TENSOR = np.asarray(BATCH_IMAGES[:self.batch_size], dtype=np.float32)
        LABELS_TENSOR = np.asarray(BATCH_LABELS[:self.batch_size], dtype=np.float32)

        return IMAGES_TENSOR, LABELS_TENSOR

    def on_epoch_end(self):
        """
        Do something at the end of each epoch
        """

        # shuffle the dataset, if necessary:
        self.index_list = np.arange(len(self.data_x))
        if self.shuffle == True:
            np.random.shuffle(self.index_list)
