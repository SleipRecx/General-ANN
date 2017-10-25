import tensorflow as tf
import numpy as np
from typing import List, Callable, Tuple
from TFlowtools import plot_training_error
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
        self.organize_cases(cases)

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
        self.weights = tf.Variable(np.random.uniform(weight_range[0], weight_range[1], size=(input_size, output_size)), name="weights")
        self.biases = tf.Variable(np.random.uniform(bias_range[0], bias_range[1], size=output_size), name="biases")
        self.output = activation(tf.matmul(input_tensor, self.weights) + self.biases)


class Network:
    def __init__(self, input_size: int, dimensions: List, activations: List, bias_range: Tuple, weight_range: Tuple,
                 loss_function: Callable, optimizer: Callable = tf.train.AdamOptimizer, learning_rate: float = 0.01,
                 minibatch_size: int = 50, epochs: int = 50, test_frequency: int = 10):
        self.learning_rate: float = learning_rate
        self.epochs = epochs
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.minibatch_size: int = minibatch_size
        self.input_size: int = input_size
        self.test_frequency = test_frequency
        self.layers: List = []
        self.input: Tensor = tf.placeholder(tf.float64, shape=(None, input_size), name='Input')
        self.output: Tensor = None
        self.target: Tensor = None
        self.correct_prediction = None
        self.accuracy = None
        self.build_model(dimensions, activations)
        self.error = loss_function(self.target, self.output)
        self.optimizer = optimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(loss_function(self.target, self.output), name='Backprop')
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
        self.target = tf.placeholder(tf.float64, shape=(None, dimensions[-1]), name='Target')
        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float64))

    def test_model(self, message: str, cases):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        error, accuracy = self.session.run([self.error, self.accuracy], feed_dict=feeder)
        print(message, accuracy)
        return error, accuracy

    def train_model(self, cases: CaseManager):
        training_errors = []
        validation_errors = []
        for epoch in range(self.epochs):
            for start in range(0, len(cases.train), self.minibatch_size):
                end = min(len(cases.train), start + self.minibatch_size)
                minibatch = cases.train[start:end]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                self.session.run([self.trainer, self.error, self.accuracy], feed_dict=feeder)
            if epoch % self.test_frequency == 0:
                e, a = self.test_model("Epoch: " + str(epoch) + " Training", cases.train)
                training_errors.append((epoch, e))
                e, a = self.test_model("Epoch: " + str(epoch) + " Validation", cases.validation)
                validation_errors.append((epoch, e))

        print("_" * 100)
        print("Training completed \n")
        self.test_model("Training", cases.train)
        self.test_model("Validation", cases.validation)
        self.test_model("Test", cases.test)
        plot_training_error(training_errors, validation_errors)