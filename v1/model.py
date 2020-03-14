from config import HISTORY_LENGTH  # it's a local py file: config.py
from config import MY_MOVEMENT  # it's a local py file: config.py
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


img_rows, img_cols = 240, 256
number_of_actions = len(MY_MOVEMENT)


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


def generate_complex_model():
    history_x_location_inputs = keras.Input(shape=(HISTORY_LENGTH*2, ), name="history_x_position")
    history_x_location_x = keras.layers.Dense(16, activation="sigmoid")(history_x_location_inputs)
    history_x_location_x = keras.layers.Dropout(0.2)(history_x_location_x)
    history_x_location_x = keras.layers.Dense(8, activation="sigmoid")(history_x_location_x)
    history_x_location_x = keras.layers.Dropout(0.2)(history_x_location_x)
    history_x_location_outputs = keras.layers.Dense(16, activation="sigmoid", name="history_x_position_output")(history_x_location_x)

    history_y_location_inputs = keras.Input(shape=(HISTORY_LENGTH*2, ), name="history_y_position")
    history_y_location_x = keras.layers.Dense(4, activation="sigmoid")(history_y_location_inputs)
    history_y_location_x = keras.layers.Dropout(0.2)(history_y_location_x)
    history_y_location_x = keras.layers.Dense(8, activation="sigmoid")(history_y_location_x)
    history_y_location_x = keras.layers.Dropout(0.2)(history_y_location_x)
    history_y_location_outputs = keras.layers.Dense(4, activation="sigmoid", name="history_y_position_output")(history_y_location_x)

    history_action_inputs = keras.Input(shape=(HISTORY_LENGTH*2, ), name="history_action")
    history_action_x = keras.layers.Dense(8, activation="sigmoid")(history_action_inputs)
    history_action_x = keras.layers.Dropout(0.2)(history_action_x)
    history_action_x = keras.layers.Dense(16, activation="sigmoid")(history_action_x)
    history_action_x = keras.layers.Dropout(0.2)(history_action_x)
    history_action_outputs = keras.layers.Dense(8, activation="sigmoid", name="history_action_output")(history_action_inputs)

    history_img_inputs = keras.Input(shape=(HISTORY_LENGTH, img_rows, img_cols, 3), name="history_image")
    history_img_x = keras.layers.Conv3D(16, (2, 2, 2), activation='sigmoid')(history_img_inputs)
    history_img_x = keras.layers.MaxPool3D((1, 2, 2))(history_img_x)
    history_img_x = keras.layers.Dropout(0.5)(history_img_x)
    history_img_x = keras.layers.Conv3D(8, (2, 2, 2), activation='sigmoid')(history_img_x)
    history_img_x = keras.layers.MaxPool3D((1, 2, 2))(history_img_x)
    history_img_x = keras.layers.Dropout(0.5)(history_img_x)
    history_img_x = keras.layers.Flatten()(history_img_x)
    history_img_outputs = keras.layers.Dense(16, activation='sigmoid', name="history_image_output")(history_img_x)

    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="image")
    img_x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='sigmoid')(img_inputs)
    img_x = keras.layers.MaxPool2D((2, 2))(img_x)
    img_x = keras.layers.Dropout(0.5)(img_x)
    img_x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='sigmoid')(img_x)
    img_x = keras.layers.MaxPool2D((2, 2))(img_x)
    img_x = keras.layers.Dropout(0.5)(img_x)
    img_x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='sigmoid')(img_x)
    img_x = keras.layers.MaxPool2D((2, 2))(img_x)
    img_x = keras.layers.Dropout(0.5)(img_x)
    img_x = keras.layers.Flatten()(img_x)
    img_outputs = keras.layers.Dense(64, activation='sigmoid', name="image_output")(img_x)

    x = keras.layers.concatenate([history_x_location_outputs, history_y_location_outputs, history_action_outputs, history_img_outputs, img_outputs], name="main_input")
    x = keras.layers.Dense(128, activation="tanh")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation="tanh")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(32, activation="tanh")(x)
    y = keras.layers.Dense(number_of_actions, activation="softmax", name="main_output")(x)

    model = keras.Model(
        inputs=[history_x_location_inputs, history_y_location_inputs, history_action_inputs, history_img_inputs, img_inputs],
        outputs=y
    )

    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model

# def generate_complex_model():
#    history_x_location_inputs = keras.Input(shape=(HISTORY_LENGTH*2, ), name="history_x_position")
#    history_x_location_x = keras.layers.Dense(8, activation="relu")(history_x_location_inputs)
#    history_x_location_x = keras.layers.Dropout(0.5)(history_x_location_x)
#    history_x_location_x = keras.layers.Dense(16, activation="relu")(history_x_location_x)
#    history_x_location_x = keras.layers.Dropout(0.5)(history_x_location_x)
#    history_x_location_outputs = keras.layers.Dense(8, activation="relu")(history_x_location_x)
#
#    history_y_location_inputs = keras.Input(shape=(HISTORY_LENGTH*2, ), name="history_y_position")
#    history_y_location_x = keras.layers.Dense(8, activation="relu")(history_y_location_inputs)
#    history_y_location_x = keras.layers.Dropout(0.5)(history_y_location_x)
#    history_y_location_x = keras.layers.Dense(16, activation="relu")(history_y_location_x)
#    history_y_location_x = keras.layers.Dropout(0.5)(history_y_location_x)
#    history_y_location_outputs = keras.layers.Dense(8, activation="relu")(history_y_location_x)
#
#    history_action_inputs = keras.Input(shape=(HISTORY_LENGTH*2, ), name="history_action")
#    history_action_x = keras.layers.Dense(16, activation="relu")(history_action_inputs)
#    history_action_x = keras.layers.Dropout(0.5)(history_action_x)
#    history_action_x = keras.layers.Dense(16, activation="relu")(history_action_x)
#    history_action_x = keras.layers.Dropout(0.5)(history_action_x)
#    history_action_outputs = keras.layers.Dense(32, activation="relu")(history_action_inputs)
#
#    history_img_inputs = keras.Input(shape=(HISTORY_LENGTH, img_rows, img_cols, 3), name="history_image")
#    history_img_x = keras.layers.Conv3D(16, (2, 2, 2), activation='relu')(history_img_inputs)
#    history_img_x = keras.layers.MaxPool3D((1, 2, 2))(history_img_x)
#    history_img_x = keras.layers.Dropout(0.5)(history_img_x)
#    history_img_x = keras.layers.Conv3D(16, (2, 2, 2), activation='relu')(history_img_x)
#    history_img_x = keras.layers.MaxPool3D((1, 2, 2))(history_img_x)
#    history_img_x = keras.layers.Dropout(0.5)(history_img_x)
#    history_img_x = keras.layers.Flatten()(history_img_x)
#    history_img_outputs = keras.layers.Dense(16, activation='relu')(history_img_x)
#
#    img_inputs = keras.Input(shape=(img_rows, img_cols, 3), name="image")
#    img_x = keras.layers.Conv2D(64, 4, 2, activation='relu')(img_inputs)
#    img_x = keras.layers.MaxPool2D((2, 2))(img_x)
#    img_x = keras.layers.Dropout(0.5)(img_x)
#    img_x = keras.layers.Conv2D(128, 4, 2, activation='relu')(img_x)
#    img_x = keras.layers.MaxPool2D((2, 2))(img_x)
#    img_x = keras.layers.Dropout(0.5)(img_x)
#    img_x = keras.layers.Conv2D(128, 4, 2, activation='relu')(img_x)
#    img_x = keras.layers.MaxPool2D((2, 2))(img_x)
#    img_x = keras.layers.Dropout(0.5)(img_x)
#    img_x = keras.layers.Flatten()(img_x)
#    img_outputs = keras.layers.Dense(64, activation='relu')(img_x)
#
#    x = keras.layers.concatenate([history_x_location_outputs, history_y_location_outputs, history_action_outputs, history_img_outputs, img_outputs])
#    x = keras.layers.Dense(128, activation="relu")(x)
#    x = keras.layers.Dropout(0.5)(x)
#    x = keras.layers.Dense(64, activation="relu")(x)
#    x = keras.layers.Dropout(0.5)(x)
#    x = keras.layers.Dense(32, activation="relu")(x)
#    y = keras.layers.Dense(number_of_actions, activation="softmax")(x)
#
#    model = keras.Model(
#        inputs=[history_x_location_inputs, history_y_location_inputs, history_action_inputs, history_img_inputs, img_inputs],
#        outputs=y
#    )
#
#    model.compile(optimizer='adam',
#                  loss="categorical_crossentropy",
#                  metrics=['accuracy'])
#
#    return model


if __name__ == "__main__":
    generate_complex_model()
