import keras
import numpy as np
from numpy.typing import NDArray
from pprint import pprint


def convert_to_ctensor(array: NDArray, filepath: str):
    """
    Convert numpy array to.ctensor file
    """
    with open(filepath, "w") as f:
        f.write(f"{len(array.shape)}\n")
        f.write(" ".join([str(dim) for dim in array.shape]))
        f.write(f"\n{array.size}\n")

        for value in array.flatten():
            f.write(f"{value}\n")


def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28), name="model_input"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu", name="export_dense_1"),
            keras.layers.Dense(10, activation="linear", name="export_dense_2"),
            keras.layers.Softmax(name="model_output"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    _ = model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=10
    )

    model_without_activation = keras.models.Model(
        inputs=model.layers[0].input, outputs=model.layers[-2].output
    )

    prediction = model_without_activation.predict(np.zeros((1, 28, 28)))

    print("Expect the following prediction when running the model from C:")
    pprint(prediction[0].tolist())
    print("Prediction:", np.argmax(prediction))

    model.save("./data/model.keras")

    for layer in model.layers:
        if layer.name.startswith("export"):
            print(f"Exporting layer: {layer.name}")
            weights = layer.get_weights()

            for i, weight in enumerate(weights):
                print(f"Exporting weight: {weight.shape}")
                convert_to_ctensor(weight, f"./data/{layer.name}_{i}.ctensor")


if __name__ == "__main__":
    train()
