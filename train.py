import keras
import numpy as np


def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28), name="model_input"),
            keras.layers.Dense(64, activation="relu", name="export_dense_1"),
            keras.layers.Dense(64, activation="relu", name="export_dense_2"),
            keras.layers.Dense(10, activation="softmax", name="export_dense_3"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    _ = model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=5
    )

    model.save("./data/model.keras")

    for layer in model.layers:
        if layer.name.startswith("export"):
            print(f"Exporting layer: {layer.name}")
            weights = layer.get_weights()

            for weight in weights:
                if len(weight.shape) == 2:
                    print(f"Exporting weight: {weight.shape}")
                    np.savetxt(f"./data/{layer.name}.weight", weight)
                else:
                    print(f"Exporting bias: {weight.shape}")
                    np.savetxt(f"./data/{layer.name}.bias", weight)


if __name__ == "__main__":
    train()
