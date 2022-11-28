# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/6/3 23:51
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/6/3 23:51
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import warnings

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Test TF")
parser.add_argument('--w_download', default=None, type=str, help='path to download model')
parser.add_argument('--train_data', default=None, type=str, help='path to train_data')
parser.add_argument('--train_label', default=None, type=str, help='path to train_label')
parser.add_argument('--cal_data', default=None, type=str, help='path to cal_data')
parser.add_argument('--w_upload', default=None, type=str, help='path to upload model')
parser.add_argument('--cal_result', default=None, type=str, help='path to calculation result')
parser.add_argument('--is_train', default=False, type=str, help='is to train?')
parser.add_argument('--is_cal', default=False, type=str, help='is to calculate?')

if __name__ == "__main__":
    opt = parser.parse_args()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(5, 5),
                activation='relu',
                input_shape=(28, 28, 1),
            ),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.set_weights(np.load(opt.w_download,))
    if opt.is_train:
        train_data = np.load(opt.train_data)

        train_label = np.load(opt.train_label)

        model.fit(
            train_data,
            train_label,
            batch_size=128,
            epochs=1,
            verbose=1
        )
        np.save(opt.w_upload,model.get_weights())

    if opt.is_cal:
        cal_data = np.load(opt.cal_data)
        cal_result = model.predict(
            cal_data
        )
        np.save(opt.cal_result, cal_result)
