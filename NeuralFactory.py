import json
from typing import Callable, List, Tuple
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from NeuralNetwork import Network, CaseManager
from TFlowtools import gen_all_parity_cases, gen_all_one_hot_cases, gen_vector_count_cases, gen_segmented_vector_cases


# Mean sqaured error loss function
def mean_squared_error(target, output):
    return tf.reduce_mean(tf.square(target - output), name='MSE')


# Categorical cross entropy function
def cross_entropy(target, output):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output), name='CE')


# Leaky relu activation function, currently not supported by tensorflow
def lrelu(x):
    return tf.maximum(x, 0.01 * x)


# Factory that takes json file as input and creates neural network from json config
def neural_network_factory(filename: str) -> Tuple:
    file = open(filename)
    config = json.load(file)

    """ Dataset """
    dataset_name: str = config["dataset"]["name"]
    case_fraction: float = config["dataset"]["case_fraction"]
    validation_fraction: float = config["dataset"]["validation_fraction"]
    test_fraction: float = config["dataset"]["test_fraction"]
    cases = dataset_factory(dataset_name)

    """ Arcitechture """
    input_size: int = config["arcitechture"]["input_size"]
    layer_specification: List = config["arcitechture"]["layer_specification"]
    weight_range = (config["arcitechture"]["weight_range"]["from"], config["arcitechture"]["weight_range"]["to"])
    bias_range = (config["arcitechture"]["bias_range"]["from"], config["arcitechture"]["bias_range"]["to"])
    activation_functions = list(map(lambda x: activation_factory(x), config["arcitechture"]["activation_functions"]))

    """ Training """
    optimizer: Callable = optimizer_factory(config["training"]["optimizer"])
    epochs = config["training"]["epochs"]
    minibatch_size = config["training"]["minibatch_size"]
    loss_function = loss_factory(config["training"]["loss_function"])
    learning_rate = config["training"]["learning_rate"]
    test_frequency = config["training"]["test_frequency"]

    """ Visualization """
    visualization_on = config["visualization"]["on"]
    display_weights = config["visualization"]["display_weights"]
    display_biases = config["visualization"]["display_biases"]
    display_layers = config["visualization"]["display_layers"]
    map_size = config["visualization"]["map_batch_size"]
    dendrogram_layers = config["visualization"]["dendrogram_layers"]

    # Create network
    network = Network(
        input_size=input_size,
        dimensions=layer_specification,
        activations=activation_functions,
        loss_function=loss_function,
        optimizer=optimizer,
        learning_rate=learning_rate,
        minibatch_size=minibatch_size,
        epochs=epochs,
        weight_range=weight_range,
        bias_range=bias_range,
        test_frequency=test_frequency,
        display_weights=display_weights,
        display_layers=display_layers,
        display_biases=display_biases,
        visualization_on=visualization_on,
        dendrogram_layers=dendrogram_layers,
        map_size=map_size
    )

    # Create casemanager
    cases = CaseManager(
        cases=cases,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        case_fraction=case_fraction,
    )
    return network, cases


# Factory that returns activation function from string
def activation_factory(name: str) -> Callable:
    if name.lower() == "softmax":
        return tf.nn.softmax
    elif name.lower() == "sigmoid":
        return tf.nn.sigmoid
    elif name.lower() == "relu":
        return tf.nn.relu
    elif name.lower() == "lrelu":
        return lrelu
    elif name.lower() == "tanh":
        return tf.nn.tanh
    assert False


# Factory that returns a loss function based on a string
def loss_factory(name: str) -> Callable:
    if name.upper() == "CE":
        return cross_entropy
    elif name.upper() == "MSE":
        return mean_squared_error
    assert False


# Factory that returns optimizer function from string
def optimizer_factory(name: str) -> Callable:
    if name.upper() == "ADAM":
        return tf.train.AdamOptimizer
    elif name.upper() == "SGD":
        return tf.train.GradientDescentOptimizer
    elif name.upper() == "ADAGRAD":
        return tf.train.AdagradDAOptimizer
    elif name.upper() == "RMSPROP":
        return tf.train.RMSPropOptimizer
    assert False


# Factory that return dataset from string
def dataset_factory(name: str) -> List:
    if name.lower() == "wine":
        return read_dataset("data/wine.txt", ";")
    elif name.lower() == "glass":
        return read_dataset("data/glass.txt", ",")
    elif name.lower() == "yeast":
        return read_dataset("data/yeast.txt", ",")
    elif name.lower() == "parity":
        return gen_all_parity_cases(10)
    elif name.lower() == "autoencoder":
        return gen_all_one_hot_cases(8)
    elif name.lower() == "bit counter":
        return gen_vector_count_cases(500, 15)
    elif name.lower() == "segment counter":
        return gen_segmented_vector_cases(25, 2500, 0, 8)
    elif name.lower() == 'mnist':
        mnist = input_data.read_data_sets("data/mnist", one_hot=True)
        return read_mnist(mnist.train.images, mnist.train.labels)
    elif name.lower() == "iris":
        return read_dataset("data/iris.txt", ",")
    assert False


# This read mnist and cleans it to the right format
def read_mnist(examples, targets) -> List:
    result = []
    for i in range(len(examples)):
        result.append([examples[i], targets[i]])
    return result


# Reads dataset from file, normalizes it and normalizes it and creates one hot target vector
def read_dataset(path: str, seperator: str) -> List:
    dataset = list(map(lambda x: x.split(seperator), open(path).read().split("\n")))
    for i in range(0, len(dataset)):
        dataset[i] = list(map(float, dataset[i]))
    examples = []
    targets = []
    for i in range(len(dataset)):
        examples.append(dataset[i][:-1])
        targets.append(dataset[i][-1])
    classes = np.unique(targets)
    vector = [0] * classes.shape[0]
    examples = np.array(examples)
    examples = (examples - examples.min(0)) / examples.ptp(0)
    for i in range(len(targets)):
        current = np.array(vector[:])
        number = targets[i]
        index = np.where(classes == number)
        current[index] = 1
        targets[i] = current
    result = []
    for i in range(len(examples)):
        result.append([examples[i], targets[i]])
    return result
