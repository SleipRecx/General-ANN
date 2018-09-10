from NeuralFactory import neural_network_factory
import random
import tensorflow as tf
import sys
import os
import numpy as np

# Logging Verbosity
tf.logging.set_verbosity(tf.logging.ERROR)

# Make program determenistic for testing purposes, ofc remove in production.
random.seed(123)
np.random.seed(123)
tf.set_random_seed(123)


config = sys.argv[1]

# Create network and dataset from json config.
network, cases = neural_network_factory("configs/" + config + ".json")

# Train model with casemanager.
network.train_model(cases)
