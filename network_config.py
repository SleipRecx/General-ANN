import json
from network import Network, CaseManager


def map_config_to_network(filename: str) -> Network:
    file = open(filename)
    config = json.load(file)

    dataset = config["dataset"]["name"]
    case_fraction = config["dataset"]["case_fraction"]
    validation_fraction = config["dataset"]["validation_fraction"]
    test_fraction = config["dataset"]["test_fraction"]

    input_size = config["arcitechture"]["input_size"]

    network = Network()
    return network


map_config_to_network("configs/mnist.json")
