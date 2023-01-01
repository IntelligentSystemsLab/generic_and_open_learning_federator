# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/1/1 20:08
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/1/1 20:08

import tensorflow as tf


def create_cnn_for_fmnist():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3),
                activation='relu',
                input_shape=(28, 28, 1),
                padding='SAME',
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
            tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3),
                activation='relu',
                padding='SAME',
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )

model = 'create_cnn_for_fmnist'
optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)
loss = tf.keras.losses.CategoricalCrossentropy()
batch_size = 128
train_epoch = 1
library = 'tensorflow'
metrics = ["accuracy"]
