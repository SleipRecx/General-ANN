from NeuralFactory import neural_network_factory
import random
import tensorflow as tf
import numpy as np

# Make program determenistic for testing purposes, ofc remove in production.
random.seed(123)
np.random.seed(123)
tf.set_random_seed(123)

# Create network and dataset from json config.
network, cases = neural_network_factory("configs/segments.json")

# Train model with casemanager.
network.train_model(cases)
