import tensorflow as tf
from typing import List, Callable
from tensorflow.examples.tutorials.mnist import input_data
Tensor = tf.Tensor


def mean_squared_sum(target, output):
    return tf.reduce_mean(tf.square(target - output), name='MSE')


def mean_cross_entropy(target, output):
    return tf.reduce_mean(-tf.reduce_sum(target * tf.log(output), reduction_indices=[1]), name='MCE')


class Layer:
    def __init__(self, input_tensor: Tensor, input_size, output_size, activation):
        self.output_size = output_size
        self.weights = tf.Variable(tf.random_normal([input_size, output_size]), name="weights")
        self.biases = tf.Variable(tf.random_normal([output_size]), name="biases")
        self.output = activation(tf.matmul(input_tensor, self.weights) + self.biases)


class GANN:
    def __init__(self, input_size: int, layers_specification: List, loss_function, learning_rate: float = 0.1,
                 mini_batch_size: int = 100):
        self.learning_rate: float = learning_rate
        self.mini_batch_size: int = mini_batch_size
        self.loss_function: Callable = loss_function
        self.input_size = input_size
        self.layers: List = []
        self.input: Tensor = tf.placeholder(tf.float32, shape=(None, input_size), name='Input')
        self.output: Tensor = None
        self.target: Tensor = None
        self.build_model(layers_specification)

    def build_model(self, layers_specification: List):
        prev_layer_output = self.input
        prev_layer_size = self.input_size
        for i in range(0, len(layers_specification)):
            layer_size, activation_function = layers_specification[i]
            layer: Layer = Layer(prev_layer_output, prev_layer_size, layer_size, activation_function)
            self.layers.append(layer)
            prev_layer_size = layer_size
            prev_layer_output = layer.output
        self.output = prev_layer_output
        self.target = tf.placeholder(tf.float32, shape=(None, layers_specification[-1][0]), name='Target')

    def test_model(self, inputs, targets, sess):
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={self.input: inputs, self.target: targets})
        return result

    def train_model(self, inputs: List, targets: List, epochs: int = 50):
        assert len(inputs) == len(targets), 'Inputs and targets needs to be same size'
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainer = optimizer.minimize(self.loss_function(self.target, self.output), name='Backprop')
        for epoch in range(epochs):
            print('Epoch:', epoch, self.test_model(inputs, targets, sess) * 100, '%')
            for i in range(0, len(inputs), self.mini_batch_size):
                end = min(i + self.mini_batch_size, len(inputs) - 1)
                sess.run(trainer, feed_dict={self.input: inputs[i:end], self.target: targets[i:end]})
        return sess


# Create network
layers = [(10, tf.nn.softmax)]
network: GANN = GANN(784, layers, mean_cross_entropy, 0.7)


# Train network
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
sess = network.train_model(mnist.train.images, mnist.train.labels)

# Test network
print(network.test_model(mnist.test.images, mnist.test.labels, sess) * 100, '% on test')
print(network.test_model(mnist.train.images, mnist.train.labels, sess) * 100, '% on train')
