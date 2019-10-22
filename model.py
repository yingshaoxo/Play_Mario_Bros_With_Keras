import tensorflow as tf

img_rows , img_cols = 240, 256
number_of_actions = 7

def generate_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Convolution2D(128, 8, 4, input_shape=(img_rows, img_cols, 3)),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(128, 4, 2, input_shape=(img_rows, img_cols, 3)),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(256),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax),
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model
