# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/1/1 20:08
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/1/1 20:08

import tensorflow as tf
from tensorflow.keras.regularizers import l2

def create_cnn_for_gtsrb():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, kernel_size=(5, 5),
                input_shape=(32, 32, 3),
                padding='same', kernel_regularizer=l2(1e-5)
            ),
            tf.keras.layers.BatchNormalization(epsilon=1e-6),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                32, kernel_size=(5, 5),
                padding='same', kernel_regularizer=l2(1e-5)
            ),
            tf.keras.layers.BatchNormalization(epsilon=1e-6),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=2),
            tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3),
                padding='same', kernel_regularizer=l2(1e-5)
            ),
            tf.keras.layers.BatchNormalization(epsilon=1e-6),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=2),
            tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3),
                padding='same', kernel_regularizer=l2(1e-5)
            ),
            tf.keras.layers.BatchNormalization(epsilon=1e-6),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(epsilon=1e-6),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(43, activation='softmax'),
        ]
    )


model = 'create_cnn_for_gtsrb'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00075)
loss = tf.keras.losses.CategoricalCrossentropy()
batch_size = 128
train_epoch = 1
library = 'tensorflow'
metrics = ["accuracy"]
