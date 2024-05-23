# MNIST in C

## Neural Networks + Raylib

MNIST Inference in pure C: A toy example app that lets you draw digits and predicts which number you drew using a neural network trained with TensorFlow. The inference code uses my other toy project [ctensor](https://github.com/am1tyadav/ctensor)

## Usage

Run all tasks with

```bash
task all
```

or one by one:

```bash
# setup conda environment for model training
task conda
# train the neural network on MNIST data
task train
# build the app
task build
# run the app
task run
```

## Dependencies

- [Task](https://taskfile.dev) (Optional). If you don't want to use Task, you can see the list of raw commands in the [Taskfile](Taskfile.yml)
- Conda or preferably via [Miniforge](https://github.com/conda-forge/miniforge)
- CMake
- C compiler
