##
# ingroup: ML_Utilities
# file:    sample_training.py
# brief:   This file contains a sample for using the generator to train a network
# author:  Nikita Kovalenko (mykyta.kovalenko@hhi.fraunhofer.de)
# date:    28.09.2020
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
import os, sys
import numpy as np
import time, datetime
import cv2
import json
import traceback
#----------------------
from utils.utils import *
from keras_stuff.data_generator import DataGenerator
#----------------------
from tensorflow.python.client import device_lib
#----------------------
from keras.applications.mobilenet import preprocess_input as preprocessing
from keras.utils import multi_gpu_model
#----------------------
import keras

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.initializers import *
from keras.metrics import *
from keras.losses import *
#----------------------

def main():
    # ----------------------------------
    # generate a small dataset here, with two types of images:
    # - one, containing circles,
    # - the other containing rectangles
    dataset_size = 200
    image_size = (256,256)
    classes = ['circle', 'rectangle']

    dataset = []
    labels = []

    __generate_dataset(dataset, labels, image_size, dataset_size)

    # ----------------------------------
    # create a model
    model = create_model(shape=(64,64,3), classes=len(classes))

    # ----------------------------------
    # parallelize the model (if possible):
    NUM_OF_GPUS = len([gpu for gpu in get_available_gpus() if 'gpu' in gpu.lower() and 'xla' not in gpu.lower()])
    log(f"This PC has {NUM_OF_GPUS} GPUs available")

    batch_size = 32
    train_batch_size = batch_size * NUM_OF_GPUS
    log(f"Using batch size {train_batch_size} ({batch_size} x {NUM_OF_GPUS}), divided between {NUM_OF_GPUS} GPUs")

    if NUM_OF_GPUS > 1:
        parallel_model = multi_gpu_model(model, gpus=NUM_OF_GPUS, cpu_merge=False)
        log(f"Model parallelized between {NUM_OF_GPUS} GPUs")
    else:
        parallel_model = model

    # ----------------------------------
    # create optimizer:
    momentum = 0.937
    weight_decay = 5e-4
    base_lr = 0.001
    opt = Adam(lr=base_lr, decay=weight_decay, beta_1=momentum, beta_2=0.999)
    # opt = SGD(lr=base_lr, decay=weight_decay, momentum=momentum, nesterov=True)

    # ----------------------------------
    # create loss and metric:
    loss = keras.losses.binary_crossentropy
    metric = keras.metrics.mean_squared_error

    # ----------------------------------
    # reload the model checkpoint (if possible)
    model_checkpoint = "checkpoint.h5"
    if os.path.exists(model_checkpoint):
        try:
            parallel_model.load_weights(model_checkpoint)
        except:
            log("Unable to load the checkpoint, proceeding with a fresh model")

    # ----------------------------------
    # compile the model
    parallel_model.compile(optimizer=opt, loss=loss, metrics=[metric])

    # ----------------------------------
    # reload the optimizer checkpoint (if possible)
    optimizer_checkpoint = "optimizer.dat"
    if os.path.exists(optimizer_checkpoint):
        op = open(optimizer_checkpoint, "r").read()
        op = op[op.find('{'):]
        op = op[:op.rfind('}')+1]
        op = json.loads(op)

        try:
            parallel_model.optimizer = parallel_model.optimizer.from_config(op)
        except:
            log("Unable to reload the optimizer, proceeding with a fresh one")

    # ----------------------------------
    # create model callbacks:

    ## best checkpoint
    best_checkpoint_filename = "model_best.h5"
    bestpoint = ModelCheckpoint(best_checkpoint_filename, monitor='loss', verbose=1,
                            save_best_only=True, mode='auto', save_weights_only=True)

    ## terminate, if loss is NaN
    term_on_nan = TerminateOnNaN()

    ## early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    ## optimizer checkpoint:
    class OptimizerSaver(Callback):
        def __init__(self, filename):
            super().__init__()
            self.filename = filename

        def on_epoch_end(self, epoch, logs=None):
            with open(self.filename, "w+") as f:
                f.write(f"time: {datetime.datetime.now().time()}\nepoch: {epoch}\nconfig: {json.dumps(self.model.optimizer.get_config())}\n")

    optimizer_checkpoint_filename = "optimizer.dat"
    write_optimizer = OptimizerSaver(optimizer_checkpoint_filename)

    ## list of callbacks
    callbacks_list=[
        bestpoint,
        term_on_nan,
        early_stopping,
        write_optimizer
    ]

    # ----------------------------------
    # Generator
    gen_params = {'batch_size': train_batch_size,
                  'input_size': (64,64),
                  'padding': True,
                  'one_hot_encode': True,
                  'augment':True,
                  'probability': 0.5,
                  'preprocessing': preprocessing, # (normalize to [-1..+1])
                  'shuffle': True}

    train_gen = DataGenerator(dataset, labels, **gen_params)

    # ----------------------------------
    # Start training
    start_timestamp = datetime.datetime.now()
    try:
        history = parallel_model.fit_generator(
            generator=train_gen,       # data loader/generator
            epochs=100,                # total number of epochs
            callbacks=callbacks_list,  # list of callbacks
            verbose=1,
            initial_epoch=0,           # start/continue from epoch number _
            use_multiprocessing=False, # set to False, if data generating doesn't work
            workers=4,
        )

        log(f'Model training complete at: {datetime.datetime.now()}')

        time_string = strfdelta(datetime.datetime.now() - start_timestamp, "{d} days, {h} hours, {m} minutes and {s} seconds")
        log(f"Training took {time_string}")
    except Exception:
        log("Training error")
        traceback.print_exc()

    # ----------------------------------
    # Save model:
    model_timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    model_name = "test_model"
    model_save_name = "{}_{}".format(model_name, model_timestamp)
    model_save_dir = "./models/{}".format(model_name)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    open(os.path.join(model_save_dir, model_save_name + ".json"), "w").write(model.to_json(indent=2))
    model.save_weights(os.path.join(model_save_dir, model_save_name + ".h5"))

    log(f"Model saved as '{os.path.join(model_save_dir, model_save_name)}'")


#----------------------------------------------------------------------------------------
def strfdelta(tdelta, fmt):
    d = {"d": tdelta.days}
    d["h"], rem = divmod(tdelta.seconds, 3600)
    d["m"], d["s"] = divmod(rem, 60)
    return fmt.format(**d)
#----------------------------------------------------------------------------------------
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if 'GPU' in x.device_type]
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
def create_model(shape=(64,64,3), classes=2):

    input_layer = Input(shape=shape, name="input_layer")

    x = Conv2D(64, 3, padding='same', activation='relu')(input_layer)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(classes, activation='softmax')(x)

    return Model(inputs=[input_layer], outputs=[x], name="my_pretty_model")

#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()