from config import MY_MOVEMENT  # it's a local py file: config.py
from tensorflow import keras
import os


img_rows, img_cols = 240, 256
number_of_actions = len(MY_MOVEMENT)


def generate_model():
    base_model = keras.applications.MobileNetV2(input_shape=(img_rows, img_cols, 3),
                                                include_top=False,
                                                weights='imagenet'
                                                )
    base_model.trainable = False
    image_x = keras.layers.Flatten()(base_model.outputs[0])
    image_x = keras.layers.Dense(1024, activation='sigmoid')(image_x)
    image_x = keras.layers.Dropout(0.5)(image_x)
    image_x = keras.layers.Dense(512, activation='sigmoid')(image_x)
    image_x = keras.layers.Dropout(0.5)(image_x)
    image_x = keras.layers.Dense(128, activation='sigmoid')(image_x)
    image_x = keras.layers.Dropout(0.5)(image_x)
    image_outputs = keras.layers.Dense(number_of_actions, activation='softmax', name="image_outputs")(image_x)

    model = keras.Model(
        inputs=base_model.input,
        outputs=image_outputs
    )

    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    generate_model()
