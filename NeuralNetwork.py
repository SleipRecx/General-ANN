import tensorflow as tf
import numpy as np
from typing import List, Callable, Tuple
import math
import matplotlib.pyplot as plt
from TFlowtools import plot_training_error, hinton_plot, dendrogram, bits_to_str
Tensor = tf.Tensor


class CaseManager:
    def __init__(self,
                 cases: List,
                 case_fraction: float = 1.0,
                 validation_fraction: float = 0.1,
                 test_fraction: float = 0.1):
        self.train_fraction: float = 1 - (validation_fraction + test_fraction)
        self.case_fraction = case_fraction
        self.validation_fraction: float = validation_fraction
        self.test_fraction: float = test_fraction
        self.train: List[Tuple] = None
        self.validation: List[Tuple] = None
        self.test: List[Tuple] = None
        case_count = int(len(cases) * case_fraction)
        self.organize_cases(cases[0:case_count])

    def organize_cases(self, cases: List):
        ca = np.array(cases)
        np.random.shuffle(ca)
        separator1 = round(len(cases) * self.train_fraction)
        separator2 = separator1 + round(len(cases) * self.validation_fraction)
        self.train = ca[0:separator1]
        self.validation = ca[separator1:separator2]
        self.test = ca[separator2:]


class Layer:
    def __init__(self, input_tensor: Tensor, input_size, output_size, activation, weight_range, bias_range):
        self.output_size = output_size
        self.weights = tf.Variable(np.random.uniform(weight_range[0], weight_range[1], size=(input_size, output_size)))
        self.biases = tf.Variable(np.random.uniform(bias_range[0], bias_range[1], size=output_size), name="biases")
        self.output = activation(tf.matmul(input_tensor, self.weights) + self.biases)


class Network:
    def __init__(self, input_size: int, dimensions: List, activations: List, bias_range: Tuple, weight_range: Tuple,
                 loss_function: Callable, dendrogram_layers: List, display_weights: List, display_biases: List,
                 display_layers: List, map_size: int, optimizer: Callable = tf.train.AdamOptimizer,
                 learning_rate: float = 0.01, minibatch_size: int = 50, epochs: int = 50, test_frequency: int = 10,
                 visualization_on: bool = False):
        self.learning_rate: float = learning_rate
        self.epochs = epochs
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.dendrogram_layers = dendrogram_layers
        self.map_size = map_size
        self.visualization_on = visualization_on
        self.display_weights = display_weights
        self.display_biases = display_biases
        self.display_layers = display_layers
        self.minibatch_size: int = minibatch_size
        self.input_size: int = input_size
        self.test_frequency = test_frequency
        self.layers: List = []
        self.input: Tensor = tf.placeholder(tf.float64, shape=(None, input_size), name='input')
        self.output: Tensor = None
        self.target: Tensor = None
        self.correct_prediction = None
        self.accuracy = None
        self.build_model(dimensions, activations)
        self.error = loss_function(self.target, self.output)
        self.optimizer = optimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(loss_function(self.target, self.output), name='backprop')
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def build_model(self, dimensions: List, activations: List):
        assert len(dimensions) == len(activations), 'dimensions and activations need to be same size'
        prev_layer_output = self.input
        prev_layer_size = self.input_size
        for i in range(0, len(dimensions)):
            layer = Layer(
                prev_layer_output,
                prev_layer_size,
                dimensions[i],
                activations[i],
                self.weight_range,
                self.bias_range)
            self.layers.append(layer)
            prev_layer_size = dimensions[i]
            prev_layer_output = layer.output
        self.output = prev_layer_output
        self.target = tf.placeholder(tf.float64, shape=(None, dimensions[-1]), name='target')
        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float64))

    def test_model(self, message: str, cases):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        error, accuracy = self.session.run([self.error, self.accuracy], feed_dict=feeder)
        print(message, accuracy)
        return error, accuracy

    def visualize_model(self, cases: CaseManager):
        inputs = [c[0] for c in cases.test[0:self.map_size]]
        targets = [c[1] for c in cases.test[0:self.map_size]]
        feeder = {self.input: inputs, self.target: targets}
        layers = []
        weights = []
        biases = []
        for weight in self.display_weights:
            weights.append(self.layers[weight - 1].weights)
        for bias in self.display_biases:
            biases.append(self.layers[bias - 1].biases)
        for layer in self.display_layers:
            if layer == 0:
                layers.append(self.input)
            else:
                layers.append(self.layers[layer - 1].output)

        grabbed = self.session.run([layers, weights, biases], feed_dict=feeder)
        g_layers, g_weights, g_biases = grabbed

        for i in range(len(g_weights)):
            current = g_weights[i]
            layer = self.display_weights[i]
            if layer == (len(self.layers)):
                hinton_plot(matrix=current, title="Output Layer Weights")
            else:
                hinton_plot(matrix=current, title="Layer " + str(layer) + " Weights")

        for i in range(len(g_biases)):
            current = np.array([g_biases[i]])
            layer = self.display_biases[i]
            if layer == (len(self.layers)):
                hinton_plot(matrix=current, title="Output Layer Biases")
            else:
                hinton_plot(matrix=current, title="Layer " + str(layer) + " Biases")

        for i in range(len(g_layers)):
            current = g_layers[i]
            layer = self.display_layers[i]
            if layer == 0:
                hinton_plot(matrix=current, title="Network Input")
            elif layer == (len(self.layers)):
                hinton_plot(matrix=current, title="Network Output")
            else:
                hinton_plot(matrix=current, title="Layer " + str(layer) + " Output")
        for layer in self.dendrogram_layers:
            dendrogram(g_layers[layer], list(map(lambda x: bits_to_str(x), targets)))

    def train_model(self, cases: CaseManager):
        training_errors = []
        validation_errors = []
        minibatch_numbers = math.ceil(len(cases.train) / self.minibatch_size)
        for epoch in range(self.epochs):
            error = 0
            accuracy = 0
            for start in range(0, len(cases.train), self.minibatch_size):
                end = min(len(cases.train), start + self.minibatch_size)
                minibatch = cases.train[start:end]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _, e, a = self.session.run([self.trainer, self.error, self.accuracy], feed_dict=feeder)
                error += e
                accuracy += a
            error = error / minibatch_numbers
            accuracy = accuracy / minibatch_numbers
            if epoch % self.test_frequency == 0:
                print("Epoch: " + str(epoch) + " Training", accuracy)
                training_errors.append((epoch, error))
                e, a = self.test_model("Epoch: " + str(epoch) + " Validation", cases.validation)
                validation_errors.append((epoch, e))
                print()
        print("Training completed")
        print("-" * 60)
        self.test_model("Training", cases.train)
        self.test_model("Test", cases.test)
        plot_training_error(training_errors, validation_errors)
        if self.visualization_on:
            self.visualize_model(cases)
        plt.show()
