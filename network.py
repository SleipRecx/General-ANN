import tensorflow as tf
import numpy as np
import math
from typing import List, Callable, Tuple
from tensorflow.examples.tutorials.mnist import input_data
from tflowtools import plot_training_error
Tensor = tf.Tensor


def squared_error(target, output):
    return tf.reduce_mean(tf.square(target - output), name='MSE')


def cross_entropy(target, output):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output), name='CE')


class CaseManager:
    def __init__(self, examples: List, targets: List, validation_fraction: float = 0.1, test_fraction: float = 0.1):
        self.train_fraction: float = 1 - (validation_fraction + test_fraction)
        self.validation_fraction: float = validation_fraction
        self.test_fraction: float = test_fraction
        self.train_cases: List[Tuple] = None
        self.validation_cases: List[Tuple] = None
        self.test_cases: List[Tuple] = None
        self.organize_cases(examples, targets)

    def organize_cases(self, examples, targets):
        cases: List[Tuple] = []
        for i in range(len(examples)):
            case = (np.array(examples[i]), np.array(targets[i]))
            cases.append(case)
        np.random.shuffle(cases)
        train_index: int = round(len(cases) * self.train_fraction)
        validation_index: int = train_index + round(len(cases) * self.validation_fraction)
        self.train_cases = cases[0:train_index]
        self.validation_cases = cases[train_index:validation_index]
        self.test_cases = cases[validation_index:]


class Layer:
    def __init__(self, input_tensor: Tensor, input_size, output_size, activation):
        self.output_size = output_size
        self.weights = tf.Variable(tf.random_normal([input_size, output_size]), name="weights")
        self.biases = tf.Variable(tf.random_normal([output_size]), name="biases")
        self.output = activation(tf.matmul(input_tensor, self.weights) + self.biases)


class Network:
    def __init__(self, input_size: int, dimensions: List, activations: List, loss_function,
                 optimizer: Callable = tf.train.AdamOptimizer, learning_rate: float = 0.01, mini_batch_size: int = 50):
        self.learning_rate: float = learning_rate
        self.mini_batch_size: int = mini_batch_size
        self.input_size: int = input_size
        self.layers: List = []
        self.input: Tensor = tf.placeholder(tf.float32, shape=(None, input_size), name='Input')
        self.output: Tensor = None
        self.target: Tensor = None
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
            layer = Layer(prev_layer_output, prev_layer_size, dimensions[i], activations[i])
            self.layers.append(layer)
            prev_layer_size = dimensions[i]
            prev_layer_output = layer.output
        self.output = prev_layer_output
        self.target = tf.placeholder(tf.float32, shape=(None, dimensions[-1]), name='Target')

    def train_model(self, inputs: List, targets: List, epochs: int = 100):
        assert len(inputs) == len(targets), 'Inputs and targets needs to be same size'
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
        training_errors = []
        validation_errors = []
        for epoch in range(epochs):
            error = 0
            number_of_minibatch = math.ceil(len(inputs) / self.mini_batch_size)
            for i in range(0, len(inputs), self.mini_batch_size):
                end = min(i + self.mini_batch_size, len(inputs) - 1)
                feeder = {self.input: inputs[i:end], self.target: targets[i:end]}
                _, e, a = self.session.run([self.trainer, self.error, accuracy], feed_dict=feeder)
                error += e
            error = error / number_of_minibatch
            feeder = {self.input: mnist.validation.images, self.target: mnist.validation.labels}
            e, a = self.session.run([self.error, accuracy], feed_dict=feeder)
            print(a)
            training_errors.append((epoch, error))
            validation_errors.append((epoch, e))
        print("Training completed")
        plot_training_error(training_errors, validation_errors)


"""
mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
examples = np.concatenate((mnist.train.images, mnist.validation.images, mnist.test.images), axis=0)
targets = np.concatenate((mnist.train.labels, mnist.validation.labels, mnist.test.labels), axis=0)
es = []
tes = []
for e in examples:
    es.append(e)
for t in targets:
    tes.append(t)
case_manager = CaseManager(es, tes, 0.1, 0.1)

# Create network
network: Network = Network(784, [10], [tf.nn.softmax], cross_entropy, tf.train.AdamOptimizer, 0.001, 100)

# Train network
network.train_model(mnist.train.images, mnist.train.labels)

"""