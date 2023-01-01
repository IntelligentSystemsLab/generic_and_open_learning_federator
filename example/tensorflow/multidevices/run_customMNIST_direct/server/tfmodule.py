import tensorflow as tf


def create_cnn_for_mnist():
    return tf.keras.Sequential(
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


model = 'create_cnn_for_mnist'
optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)
loss = tf.keras.losses.CategoricalCrossentropy()
batch_size = 128
train_epoch = 1
library = 'tensorflow'
metrics = ["accuracy"]
