from tensorflow import keras
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

img_rows, img_cols = 240, 256
number_of_actions = 7


def generate_model():
    model = keras.models.Sequential([
        keras.layers.Convolution2D(32, 8, 4, input_shape=(img_rows, img_cols, 3)),
        keras.layers.Activation('relu'),

        keras.layers.Flatten(),
        keras.layers.Dense(256),
        keras.layers.Activation('relu'),

        keras.layers.Dense(number_of_actions, activation="softmax"),
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model


"""
def generate_complex_model():
    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="img")
    img_x_middle = keras.layers.Conv2D(128, 8, 4, activation='relu')(img_inputs)
    img_x = keras.layers.MaxPool2D(2)(img_x_middle)
    img_x = keras.layers.Conv2D(32, 4, 2, activation='relu')(img_x)
    img_x = keras.layers.Flatten()(img_x)
    img_outputs = keras.layers.Dense(128, activation='relu')(img_x)

    special_img = keras.layers.Conv2D(32, 5)(img_x_middle)
    special_img = keras.layers.LeakyReLU(0.2)(special_img)
    special_img = keras.layers.MaxPool2D(2)(special_img)
    special_img = keras.layers.Conv2D(64, 3)(special_img)
    special_img = keras.layers.LeakyReLU(0.2)(special_img)
    special_img_outputs = keras.layers.Dense(32, activation='relu')(keras.layers.Flatten()(special_img))

    part_img = keras.layers.Conv2D(32, 5, 2)(special_img)
    part_img = keras.layers.LeakyReLU(0.2)(part_img)
    part_img = keras.layers.SpatialDropout2D(0.1)(part_img)
    part_img = keras.layers.Conv2D(64, 3, 2)(part_img)
    part_img = keras.layers.LeakyReLU(0.2)(part_img)
    part_img = keras.layers.SpatialDropout2D(0.1)(part_img)
    part_img = keras.layers.Conv2D(64, 3, 2)(part_img)
    part_img = keras.layers.LeakyReLU(0.2)(part_img)
    part_img = keras.layers.Dropout(0.2)(part_img)
    part_img = keras.layers.Flatten()(part_img)
    part_img_outputs = keras.layers.Dense(32, activation='relu')(part_img)

    history_action_inputs = keras.Input(shape=(100, ), name="action")
    history_action_x = keras.layers.Dense(2)(history_action_inputs)
    history_action_outputs = keras.layers.Dense(64, activation="relu")(history_action_x)

    history_x_location_inputs = keras.Input(shape=(100, ), name="x_position")
    history_x_location_x = keras.layers.Dense(16)(history_x_location_inputs)
    history_x_location_outputs = keras.layers.Dense(8, activation="relu")(history_x_location_x)

    history_y_location_inputs = keras.Input(shape=(100, ), name="y_position")
    history_y_location_x = keras.layers.Dense(16)(history_y_location_inputs)
    history_y_location_outputs = keras.layers.Dense(8, activation="relu")(history_y_location_x)

    x = keras.layers.concatenate([img_outputs, special_img_outputs, part_img_outputs, history_action_outputs, history_x_location_outputs, history_y_location_outputs])
    x = keras.layers.Dense(512, activation="relu")(x)
    y = keras.layers.Dense(number_of_actions, activation="softmax")(x)

    model = keras.Model(
        inputs=[img_inputs, history_action_inputs, history_x_location_inputs, history_y_location_inputs],
        outputs=y
    )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
"""


"""
def generate_complex_model():
    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="img")
    img_x = keras.layers.Conv2D(256, 5)(img_inputs)
    img_x = keras.layers.LeakyReLU(0.2)(img_x)
    img_x = keras.layers.MaxPool2D(2)(img_x)
    #img_x = keras.layers.Conv2D(64, 3)(img_x)
    #img_x = keras.layers.LeakyReLU(0.2)(img_x)
    img_x = keras.layers.Flatten()(img_x)
    img_outputs = keras.layers.Dense(64, activation='relu')(img_x)

    history_action_inputs = keras.Input(shape=(32, ), name="action")
    history_action_x = keras.layers.Dense(2)(history_action_inputs)
    history_action_outputs = keras.layers.Dense(32, activation="relu")(history_action_x)

    history_x_location_inputs = keras.Input(shape=(32, ), name="x_position")
    history_x_location_x = keras.layers.Dense(16)(history_x_location_inputs)
    history_x_location_outputs = keras.layers.Dense(8, activation="relu")(history_x_location_x)

    history_y_location_inputs = keras.Input(shape=(32, ), name="y_position")
    history_y_location_x = keras.layers.Dense(16)(history_y_location_inputs)
    history_y_location_outputs = keras.layers.Dense(8, activation="relu")(history_y_location_x)

    x = keras.layers.concatenate([img_outputs, history_action_outputs, history_x_location_outputs, history_y_location_outputs])
    x = keras.layers.Dense(256, activation="relu")(x)
    y = keras.layers.Dense(number_of_actions, activation="softmax")(x)

    model = keras.Model(
        inputs=[img_inputs, history_action_inputs, history_x_location_inputs, history_y_location_inputs],
        outputs=y
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  #loss_weights=[1., 0.5, 0.2, 0.2],
                  metrics=['accuracy'])

    return model
"""


def generate_complex_model():
    action_inputs = keras.Input(shape=(1, ), name="action")
    action_outputs = keras.layers.Dense(8, activation="relu")(action_inputs)

    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="img")
    img_x = keras.layers.Conv2D(64, 8, 4, activation='relu')(img_inputs)
    img_x = keras.layers.MaxPool2D(2)(img_x)
    img_x = keras.layers.Conv2D(32, 4, 2, activation='relu')(img_x)
    img_x = keras.layers.Flatten()(img_x)
    img_outputs = keras.layers.Dense(128, activation='relu')(img_x)

    history_action_inputs = keras.Input(shape=(100, ), name="history_action")
    history_action_outputs = keras.layers.Dense(32, activation="relu")(history_action_inputs)

    history_x_location_inputs = keras.Input(shape=(100, ), name="history_x_position")
    history_x_location_outputs = keras.layers.Dense(32, activation="relu")(history_x_location_inputs)

    history_y_location_inputs = keras.Input(shape=(100, ), name="history_y_position")
    history_y_location_outputs = keras.layers.Dense(32, activation="relu")(history_y_location_inputs)

    x = keras.layers.concatenate([action_outputs, img_outputs, history_action_outputs, history_x_location_outputs, history_y_location_outputs])
    x = keras.layers.Dense(512, activation="relu")(x)
    y = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(
        inputs=[action_inputs, img_inputs, history_action_inputs, history_x_location_inputs, history_y_location_inputs],
        outputs=y
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    generate_complex_model()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

img_rows, img_cols = 240, 256
number_of_actions = 7


def generate_model():
    model = keras.models.Sequential([
        keras.layers.Convolution2D(32, 8, 4, input_shape=(img_rows, img_cols, 3)),
        keras.layers.Activation('relu'),

        keras.layers.Flatten(),
        keras.layers.Dense(256),
        keras.layers.Activation('relu'),

        keras.layers.Dense(number_of_actions, activation="softmax"),
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model


"""
def generate_complex_model():
    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="img")
    img_x_middle = keras.layers.Conv2D(128, 8, 4, activation='relu')(img_inputs)
    img_x = keras.layers.MaxPool2D(2)(img_x_middle)
    img_x = keras.layers.Conv2D(32, 4, 2, activation='relu')(img_x)
    img_x = keras.layers.Flatten()(img_x)
    img_outputs = keras.layers.Dense(128, activation='relu')(img_x)

    special_img = keras.layers.Conv2D(32, 5)(img_x_middle)
    special_img = keras.layers.LeakyReLU(0.2)(special_img)
    special_img = keras.layers.MaxPool2D(2)(special_img)
    special_img = keras.layers.Conv2D(64, 3)(special_img)
    special_img = keras.layers.LeakyReLU(0.2)(special_img)
    special_img_outputs = keras.layers.Dense(32, activation='relu')(keras.layers.Flatten()(special_img))

    part_img = keras.layers.Conv2D(32, 5, 2)(special_img)
    part_img = keras.layers.LeakyReLU(0.2)(part_img)
    part_img = keras.layers.SpatialDropout2D(0.1)(part_img)
    part_img = keras.layers.Conv2D(64, 3, 2)(part_img)
    part_img = keras.layers.LeakyReLU(0.2)(part_img)
    part_img = keras.layers.SpatialDropout2D(0.1)(part_img)
    part_img = keras.layers.Conv2D(64, 3, 2)(part_img)
    part_img = keras.layers.LeakyReLU(0.2)(part_img)
    part_img = keras.layers.Dropout(0.2)(part_img)
    part_img = keras.layers.Flatten()(part_img)
    part_img_outputs = keras.layers.Dense(32, activation='relu')(part_img)

    history_action_inputs = keras.Input(shape=(100, ), name="action")
    history_action_x = keras.layers.Dense(2)(history_action_inputs)
    history_action_outputs = keras.layers.Dense(64, activation="relu")(history_action_x)

    history_x_location_inputs = keras.Input(shape=(100, ), name="x_position")
    history_x_location_x = keras.layers.Dense(16)(history_x_location_inputs)
    history_x_location_outputs = keras.layers.Dense(8, activation="relu")(history_x_location_x)

    history_y_location_inputs = keras.Input(shape=(100, ), name="y_position")
    history_y_location_x = keras.layers.Dense(16)(history_y_location_inputs)
    history_y_location_outputs = keras.layers.Dense(8, activation="relu")(history_y_location_x)

    x = keras.layers.concatenate([img_outputs, special_img_outputs, part_img_outputs, history_action_outputs, history_x_location_outputs, history_y_location_outputs])
    x = keras.layers.Dense(512, activation="relu")(x)
    y = keras.layers.Dense(number_of_actions, activation="softmax")(x)

    model = keras.Model(
        inputs=[img_inputs, history_action_inputs, history_x_location_inputs, history_y_location_inputs],
        outputs=y
    )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
"""


"""
def generate_complex_model():
    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="img")
    img_x = keras.layers.Conv2D(256, 5)(img_inputs)
    img_x = keras.layers.LeakyReLU(0.2)(img_x)
    img_x = keras.layers.MaxPool2D(2)(img_x)
    #img_x = keras.layers.Conv2D(64, 3)(img_x)
    #img_x = keras.layers.LeakyReLU(0.2)(img_x)
    img_x = keras.layers.Flatten()(img_x)
    img_outputs = keras.layers.Dense(64, activation='relu')(img_x)

    history_action_inputs = keras.Input(shape=(32, ), name="action")
    history_action_x = keras.layers.Dense(2)(history_action_inputs)
    history_action_outputs = keras.layers.Dense(32, activation="relu")(history_action_x)

    history_x_location_inputs = keras.Input(shape=(32, ), name="x_position")
    history_x_location_x = keras.layers.Dense(16)(history_x_location_inputs)
    history_x_location_outputs = keras.layers.Dense(8, activation="relu")(history_x_location_x)

    history_y_location_inputs = keras.Input(shape=(32, ), name="y_position")
    history_y_location_x = keras.layers.Dense(16)(history_y_location_inputs)
    history_y_location_outputs = keras.layers.Dense(8, activation="relu")(history_y_location_x)

    x = keras.layers.concatenate([img_outputs, history_action_outputs, history_x_location_outputs, history_y_location_outputs])
    x = keras.layers.Dense(256, activation="relu")(x)
    y = keras.layers.Dense(number_of_actions, activation="softmax")(x)

    model = keras.Model(
        inputs=[img_inputs, history_action_inputs, history_x_location_inputs, history_y_location_inputs],
        outputs=y
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  #loss_weights=[1., 0.5, 0.2, 0.2],
                  metrics=['accuracy'])

    return model
"""


def generate_complex_model():
    action_inputs = keras.Input(shape=(1, ), name="action")
    action_outputs = keras.layers.Dense(8, activation="relu")(action_inputs)

    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="img")
    img_x = keras.layers.Conv2D(64, 8, 4, activation='relu')(img_inputs)
    img_x = keras.layers.MaxPool2D(2)(img_x)
    img_x = keras.layers.Conv2D(32, 4, 2, activation='relu')(img_x)
    img_x = keras.layers.Flatten()(img_x)
    img_outputs = keras.layers.Dense(128, activation='relu')(img_x)

    history_action_inputs = keras.Input(shape=(100, ), name="history_action")
    history_action_outputs = keras.layers.Dense(32, activation="relu")(history_action_inputs)

    history_x_location_inputs = keras.Input(shape=(100, ), name="history_x_position")
    history_x_location_outputs = keras.layers.Dense(32, activation="relu")(history_x_location_inputs)

    history_y_location_inputs = keras.Input(shape=(100, ), name="history_y_position")
    history_y_location_outputs = keras.layers.Dense(32, activation="relu")(history_y_location_inputs)

    x = keras.layers.concatenate([action_outputs, img_outputs, history_action_outputs, history_x_location_outputs, history_y_location_outputs])
    x = keras.layers.Dense(512, activation="relu")(x)
    y = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(
        inputs=[action_inputs, img_inputs, history_action_inputs, history_x_location_inputs, history_y_location_inputs],
        outputs=y
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    generate_complex_model()
