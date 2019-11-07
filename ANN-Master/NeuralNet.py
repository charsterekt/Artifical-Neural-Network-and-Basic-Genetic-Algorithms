"""The following code is the exact same as that in TutorialNet. This is a comment free version for easier reading."""

import random
from math import exp, inf
from typing import List, Tuple, Iterator
from collections import namedtuple
from contextlib import suppress


class IntegrityError(Exception):
    pass


def sigmoid(input, *, derivative=False):
    if derivative:
        return sigmoid(input) * (1 - sigmoid(input))
    else:
        if input < -709:
            return 0
        elif input > 37:
            return 1
        return 1 / (1 + exp(-input))


def tanh(input, *, derivative=False):
    if derivative:
        return 1 - pow(tanh(input), 2)
    else:
        if input < -709:
            return -1
        elif input > 709:
            return 1
        return (exp(input) - exp(-input)) / (exp(input) + exp(-input))


def relu(input, derivative=False):
    if derivative:
        if input > 0:
            return 1
        elif input < 0:
            return 0
        else:
            return 0.5
    else:
        return max(0, input)


def leaky_relu(input, derivative=False):
    if derivative:
        if input > 0:
            return 1
        elif input < 0:
            return 0.01
        else:
            return 0.505
    else:
        return max(0, input) + min(0, input * 0.01)


class Neuron:
    def __init__(self, layer, previous_layer=None):
        self.output = 0
        self.layer = layer
        self.error = 0
        self.dendrons = []
        self.bias = 2 * random.random() - 1
        self.activation_function = tanh

        if previous_layer is not None:
            for neuron in previous_layer.neurons:
                con = Connection(neuron)
                self.dendrons.append(con)

    def set_output(self, output):
        self.output = output

    def set_error(self, error):
        self.error = error

    def add_error(self, error):
        self.error += error

    def set_activation_function(self, func):
        self.activation_function = func

    def feed_forward(self):
        if self.dendrons:
            sigma = self.bias
            for dendron in self.dendrons:
                sigma += dendron.get_value()
            self.output = self.activation_function(sigma)

    def back_propagate(self):
        gradient = self.error * self.activation_function(self.output, derivative=True)
        learning_rate = self.layer.network.learning_rate
        for dendron in self.dendrons:
            delta_weight = gradient * learning_rate * dendron.source_neuron.output
            dendron.adjust_weight(delta_weight)
            dendron.pass_gradient(gradient)
        self.bias += gradient * learning_rate * 1
        self.error = 0

    def export_data(self) -> List[float]:
        data = [self.bias]
        data.extend(dendron.weight for dendron in self.dendrons)
        return data

    def import_data(self, data_iter: Iterator[float]):
        self.bias = next(data_iter)  # reassigning the bias which was at the start of the exported data
        for dendron in self.dendrons:
            dendron.weight = next(data_iter)


class Connection:
    def __init__(self, source_neuron):
        self.source_neuron = source_neuron
        self.weight = 2 * random.random() - 1
        self.delta_weight = 0

    def get_value(self):
        return self.source_neuron.output * self.weight

    def adjust_weight(self, delta):
        self.delta_weight = delta + self.delta_weight * self.source_neuron.layer.network.momentum_mod
        self.weight += delta

    def pass_gradient(self, gradient):
        self.source_neuron.add_error(self.weight * gradient)


class Layer:
    def __init__(self, network, neuron_count: int, previous_layer=None):
        self.network = network
        self.neurons = [Neuron(self, previous_layer) for _ in range(neuron_count)]

    def set_inputs(self, inputs):
        if len(self.neurons) != len(inputs):
            raise IntegrityError("Incorrect number of inputs")
        for input, neuron in zip(inputs, self.neurons):
            neuron.set_output(input)

    def get_outputs(self):
        return [neuron.output for neuron in self.neurons]

    def set_errors(self, targets: List[float]):
        if len(self.neurons) != len(targets):
            raise IntegrityError("Incorrect number of outputs")
        outputs = self.get_outputs()
        errors = [(target - output) for target, output in zip(targets, outputs)]
        for neuron, error in zip(self.neurons, errors):
            neuron.set_error(error)
        return sum(e ** 2 for e in errors)

    def set_activation_function(self, func):
        for neuron in self.neurons:
            neuron.set_activation_function(func)

    def __len__(self):
        return len(self.neurons)

    def feed_forward(self):
        for neuron in self.neurons:
            neuron.feed_forward()

    def back_propagate(self):
        for neuron in self.neurons:
            neuron.back_propagate()

    def export_data(self) -> List[float]:
        data = []
        for neuron in self.neurons:
            data.extend(neuron.export_data())
        return data

    def import_data(self, data_iter: Iterator[float]):
        for neuron in self.neurons:
            neuron.import_data(data_iter)


class Network:
    def __init__(self, layer_counts: List[int], learning_rate=0.05, momentum_mod: float = 0.05):
        if len(layer_counts) < 2:
            raise IntegrityError("A network cannot have less than 2 layers")
        self.layers = []
        layer = None
        for count in layer_counts:
            layer = Layer(self, count, layer)
            self.layers.append(layer)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.learning_rate = learning_rate
        self.momentum_mod = momentum_mod
        self.fitness = 0

    def set_activation_function(self, func):
        for layer in self.layers:
            layer.set_activation_function(func)

    def feed_forward(self, inputs: List[float]):
        self.input_layer.set_inputs(inputs)
        for layer in self.layers:
            layer.feed_forward()
        return self.output_layer.get_outputs()

    def back_propagate(self, targets: List[float]):
        current_error = self.output_layer.set_errors(targets)
        for layer in reversed(self.layers):
            layer.back_propagate()
        return current_error

    def teach(self, dataset: list):
        net_error = 0
        for inputs, targets in dataset:
            self.feed_forward(inputs)
            net_error += self.back_propagate(targets)
        return net_error

    def export_data(self) -> List[float]:
        data = []
        for layer in self.layers:
            data.extend(layer.export_data())
        return data

    def import_data(self, data: List[float]):
        with suppress(TypeError):
            if len(data) != sum(len(n.dendrons)+1 for l in self.layers for n in l.neurons):
                raise IntegrityError("Invalid import data size")
        data_iter = iter(data)
        for layer in self.layers:
            layer.import_data(data_iter)

    def crossover(self, other: "Network") -> Tuple["Network", "Network"]:
        cls = type(self)
        n1 = self.export_data()
        n2 = other.export_data()
        assert len(n1) == len(n2)
        n = random.randint(1, len(n1) - 1)
        args = ([len(layer) for layer in self.layers], self.learning_rate, self.momentum_mod)
        n3 = cls(*args)
        n3.import_data(n1[:n] + n2[n:])
        n4 = cls(*args)
        n4.import_data(n2[:n] + n1[n:])
        return (n3, n4)

    def mutate(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for dendron in neuron.dendrons:
                    if random.random() < 0.05:
                        dendron.weight += dendron.weight * (0.2 * random.random() - 0.1)


# ----------------------------------------------------------------------------------------------------------------------
# Code content ends, example follows:

if __name__ == "__main__":
    Entry = namedtuple("Entry", ["inputs", "outputs"])

    dataset = [
        Entry([0, 0], [0, 0, 0]),
        Entry([0, 1], [1, 0, 1]),
        Entry([1, 0], [1, 0, 1]),
        Entry([1, 1], [1, 1, 0]),
    ]

    net = Network([2, 3, 3])

    precision = 0.0001
    network_error = inf
    while network_error > precision:
        network_error = net.teach(dataset)
        print(network_error)

    print('')
    exit_loop = False
    while True:
        inputs = []
        for i in range(len(net.input_layer)):
            inp = input("Input {}: ".format(i))
            if inp.lower() == "exit":
                exit_loop = True
                break
            while True:
                try:
                    inp = float(inp)
                    break
                except ValueError:
                    pass
            inputs.append(inp)
        if exit_loop:
            break
        outputs = net.feed_forward(inputs)
        print(outputs)
