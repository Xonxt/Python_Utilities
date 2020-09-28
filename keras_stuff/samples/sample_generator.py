##
# ingroup: ML_Utilities
# file:    sample_generator.py
# brief:   This file contains a sample for using the generator
# author:  Nikita Kovalenko (mykyta.kovalenko@hhi.fraunhofer.de)
# date:    26.09.2020
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
#----------------------
from utils.utils import *
from keras_stuff.data_generator import DataGenerator
#----------------------
from keras.applications.mobilenet import preprocess_input as preprocessing

#----------------------------------------------------------------------------------------
def main():
    # generate a small dataset here, with two types of images:
    # - one, containing circles,
    # - the other containing rectangles
    dataset_size = 200
    image_size = (256,256)
    classes = ['circle', 'rectangle']

    dataset = []
    labels = []

    __generate_dataset(dataset, labels, image_size, dataset_size)
    
    # test the generator
    gen_params = {'batch_size': 8,
                  'input_size': (368,368),
                  'padding': True,
                  'one_hot_encode': True,
                  'augment':True,
                  'probability': 0.5,
                  'preprocessing': preprocessing, # (normalize to [-1..+1])
                  'shuffle': True}

    gen = DataGenerator(dataset, labels, **gen_params)
    
    # generate a batch:
    image_batch, label_batch = next(iter(gen))

#----------------------------------------------------------------------------------------
def __generate_dataset(dataset, labels, image_size, dataset_size):
    # create circle images
    for i in range(dataset_size // 2):
        image_circle = (np.random.rand(image_size[0], image_size[1], 3) * np.random.randint(25,45)).astype(np.uint8)
        color = tuple(np.random.randint(200,255,3).tolist())
        radius = np.random.randint(image_size[0] * 0.25, image_size[1] * 0.4)
        cv2.circle(image_circle, (image_size[0]//2, image_size[1]//2), radius, color=color, thickness=-1)
        dataset.append(image_circle)
        labels.append(0)

    # create rectangle images
    for i in range(dataset_size // 2):
        image_rectangle = (np.random.rand(image_size[0], image_size[1], 3) * np.random.randint(25,45)).astype(np.uint8)
        color = tuple(np.random.randint(200,255,3).tolist())
        radius = np.random.randint(image_size[0] * 0.25, image_size[1] * 0.4, 2)
        cv2.rectangle(image_rectangle, (image_size[0]//2 - radius[0], image_size[1]//2 - radius[1]),
                    (image_size[0]//2 + radius[0], image_size[1]//2 + radius[1]), color=color, thickness=-1)
        dataset.append(image_rectangle)
        labels.append(1)
        
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()