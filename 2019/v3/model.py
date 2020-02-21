import tensorflow as tf

img_rows , img_cols = 240, 256
number_of_actions = 12

def generate_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Convolution2D(32, 8, 4, input_shape=(img_rows, img_cols, 3)),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(64),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(number_of_actions, activation=tf.nn.softmax),
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model
