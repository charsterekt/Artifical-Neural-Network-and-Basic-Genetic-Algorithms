# Here lie all our imports :)

import random
from math import exp, inf
from typing import List, Tuple, Iterator
from collections import namedtuple
from contextlib import suppress


"""This basic neural network uses the concepts of feed forward and backpropagation algos. We will also implement a 
   genetic algorithm to help the applications of these networks learn. We will be using the single point crossover
   method for this. All the elements of the genetic algorithm cannot be implemented in this core file, some of them 
   have to be locally implemented at the file where we are trying to automate.
   Refer to these to fully understand the code. Methods pertaining to each respective algo will be named
   as such within each class where they are defined."""

"""If you do not understand something in the code at one point, don't worry and move on. Chances are, you'll read a 
   later bit of code and understand what was happening before. I myself wrote this one method at a time in varying order.
   Just go on ahead and come back to it later."""

"""Here i'll be defining our own "error" so to speak. An exception that we will raise 
   later on in the code wherever we require it. do not worry about it for now, just know
   that we are defining our own exception."""


class IntegrityError(Exception):
    pass  # we don't need the class to do anything, we're just deriving from Exceptions
    # to name our own error


"""The activation function for this neural network will be one of many. Sigmoid was the first function ever used but is 
   very outdated and inefficient. Thus we will use the TanH function. But sigmoid has also been defined below for 
   reference. Now we need to define and create this function within code and include a derivative of each function
   for later use in finding gradient"""


# Just an example
def sigmoid(input, *, derivative = False):
    if derivative:
        return sigmoid(input) * (1 - sigmoid(input))
    else:
        if input < -709:
            return 0
        elif input > 37:
            return 1
        return 1 / (1 + exp(-input))


"""refer to the logistics function for this code :
   https://en.wikipedia.org/wiki/Sigmoid_function
   The reason we have such arbitrary values defined is that exp() would crash if we got too high.
   Also, required output seems to be achieved at 38 already, no need to bother going higher.
   For a simple test like this, such values are enough. We don't want an OverflowError.
   Here we use exp() for e raised to the power of <value>. It is more accurate than using
   pow(e,x) or e**x."""


# This is the TanH function:
"""http://functions.wolfram.com/ElementaryFunctions/Tanh/introductions/Tanh/ShowAll.html"""


def tanh(input, *, derivative = False):
    if derivative:
        return 1 - pow(tanh(input), 2)
    else:
        return (exp(input) - exp(-input)) / (exp(input) + exp(-input))


# Here's another function called ReLu. We don't want to use this for networks smaller than 5 layers as it excels mainly
# For deep neural networks
"""https://en.wikipedia.org/wiki/Rectifier_(neural_networks)"""


def relu(input, derivative = False):
    if derivative:
        if input > 0:
            return 1
        elif input < 0:
            return 0
        else:
            return 0.5
    else:
        return max(0, input)

# Here's another function example which is a variation on reLu called Leaky ReLu:


def leaky_relu(input, derivative = False):
    if derivative:
        if input > 0:
            return 1
        elif input < 0:
            return 0.01
        else:
            return 0.505
    else:
        return max(0, input) + min(0, input * 0.01)


"""Now we could use the function directly but tanh diverts from the standard neural network model. Generally, the range 
   of the output should be 0-1 but tanh returns output range from -1 - 1. To fix this, we could make it so tanh only
   applies on the hidden layers and sigmoid applies to the output, by doing :
   net.set_activation_f(tanh)
   net.output_layer.set_activation_f(sigmoid)
   when calling network class (as net).
   Alternatively, if we want to stick to sigmoid, we have to resolve the issue of Sigmoid settling in local minimums.
   We want the best result and thus we want it to settle in the global minimum. But when it settles in the local 
   minimum, the weight settles and prevents it from moving further down the function.
   To remedy this we can implement a "momentum" to allow the function to spill over the local minimum and attempt to 
   reach the global minimum."""


# creating and giving identity to the most basic unit, a neuron:


"""Neuron should know which layer it belongs to
   so that it can also know what was the previous layer.
   If there was no previous layer, then it's an input neuron.
   The previous layer is needed so it can make connections to all neurons from it."""


class Neuron:
    def __init__(self, layer, previous_layer=None):
        self.output = 0  # store output during feed forward.
        self.layer = layer
        self.error = 0  # hold the error sum during back propagation.
        self.dendrons = []  # each actual connection b/w neuron and all output neurons.
        self.bias = 2 * random.random() - 1  # a random range between -1 - 1 for bias
        self.activation_function = tanh  # this will be the activation function

        """a bias is simply another input from a "fake neuron" with weight = 1
           random.random() has a 0-1 range and *2 makes it 0-2, then -1 gives -1 - 1 range."""

        """in case when the previous_layer is supplied
           you have to create connections to each neuron from said layer
           and store them in self.connections"""

        if previous_layer is not None:
            for neuron in previous_layer.neurons:
                con = Connection(neuron)
                self.dendrons.append(con)

    def set_output(self, output):
        self.output = output

        """This is a helper function to take care of the neurons being input neurons;
        you have to have a way of just setting the output to whatever you want it to be"""

    def set_error(self, error):
        self.error = error

        """the next helper function would be for the backpropagation algo
           for the output neurons, to set their errors to what you want
           so you can start the algo"""

    def add_error(self, error):  # for all intermediate neurons, so they can sum up the error
        self.error += error  # +ve -ve takes care of itself

    def set_activation_function(self, func):
        self.activation_function = func

        """and finally, for the network controller (that's going to come later on),
           so that you can change the activation function easily too:"""

    # And now a part of the feed forward algo:

    def feed_forward(self):
        if self.dendrons:  # Will run if there are any dendrons, else will just skip as discussed in logic below
            sigma = self.bias
            for dendron in self.dendrons:
                sigma += dendron.get_value()
            self.output = self.activation_function(sigma)

    """Where sigma is simply the sum which we cannot name sum because it shadows an inbuilt function.
       We start with sigma = self.bias instead of sigma = 0 because according the feed forward algo, we need 
       to add bias to the sum anyway. If you understood the usage of the functions we have defined, then these lines
       of code are simply using them to calculate the appropriate output."""

    """remember that neurons can be used as input neurons
       where they don't really do anything, other than just having their output being set
       if you'd set their outputs to something and run this feed_forward method on them
       it'd take the bias as the sum
       the loop wouldn't even run because there's no dendrons connected
       so the sum / bias would go into the activation function
       and overwrite the output that you just set
       so, you need an if statement before this whole thing
       that'll check if the neuron isn't the input neuron
       because if it is, you can skip the entire feed_forward on it
       the best way to do this is to check if self.dendrons is empty or not"""

    # Now we start with the definition of back propagation algo for Neuron
    # This is what actually makes the network "Learn"

    def back_propagate(self):
        # Gradient = Error made by the neuron * output of the neuron passed into the d/dx of the activation function
        gradient = self.error * self.activation_function(self.output, derivative=True)

        """For delta_weight we need gradient, learning rate 9stored in network instance, and output of the neuron that
           weight is connected to. This delta weight has to be done on every connection, so we take a loop. But before 
           the loop we need the learning rate. If you've wondered why exactly each neuron keeps track of which layer it's 
           on and each layer keeps track of which network it's on, here's why"""

        learning_rate = self.layer.network.learning_rate
        # this will reach into the network instance the neuron is operating on, and fetch the learning rate

        # This is the loop we spoke of
        for dendron in self.dendrons:
            delta_weight = gradient * learning_rate * dendron.source_neuron.output
            dendron.adjust_weight(delta_weight)
            dendron.pass_gradient(gradient)
        # We need to make the bias learn right here as well
        self.bias += gradient * learning_rate * 1
        # There is no output from the neuron since it's 1 for bias right now.

        # And finally reset errors to 0 for the next pass
        self.error = 0

    # Now i want a method that will convert the entire network to a vector and one to convert it back
    # This will help in our single point crossover genetic algo

    # Firstly, an export function to convert the entire network into a vector (one dimensional vector is same as a list)

    def export_data(self) -> List[float]:
        data = [self.bias]  # Starting with the list containing bias only
        data.extend(dendron.weight for dendron in self.dendrons) # Now adding all the weights
        return data # And we're done

    # Now to import it. Imports were defined in reverse order so the first relevant import is in Network -> Layer and
    # Now we're here

    def import_data(self, data_iter: Iterator[float]):
        self.bias = next(data_iter)  # reassigning the bias which was at the start of the exported data
        for dendron in self.dendrons:
            dendron.weight = next(data_iter)


# A connection class that houses weights on each connecting dendron


class Connection:
    def __init__(self, source_neuron):
        self.source_neuron = source_neuron  # To represent the input as the output of the previous neuron
        self.weight = 2 * random.random() - 1  # To assign a random weight to the neuron to begin with
        self.delta_weight = 0  # This is for momentum ( as discussed before, to prevent settling into local minimas)

    def get_value(self):
        return self.source_neuron.output * self.weight

    """as a helper function for the connection, define a method
       as shown above, that will take the source neuron output,
       multiply it by the weight
       and give you the connection value;
       so that you can later easily iterate over all connections, and call that method
       to get the value of each connection"""

    def adjust_weight(self, delta):  # To adjust the weights after backpropagating so we "learn" by reducing errors.
        self.delta_weight = delta + self.delta_weight * self.source_neuron.layer.network.momentum_mod  # We have to
        # get momentum_mod from all the way over in the netwrok class, hence it's written as such
        self.weight += self.delta_weight

    # A rather simple logic that returns the new weight by adding the old to the difference
    # But we also have momentum so :
    """you have to make it so
       the delta that's passed gets added the self.delta_weight stored, multiplied by momentum_mod
       and that's assigned back to self.delta_weight, for the next operation
       and then it's added to the weight
       so like, because we're adding the stored version now, change the line to self.weight += self.delta_weight
       and then, in the line above, do an assignment to that"""

    # Now for a method that will add the error to the source neuron:

    def pass_gradient(self, gradient):
        self.source_neuron.add_error(self.weight * gradient)


# A layer controller responsible for managing groups of neurons and, it has to have a pointer to the network it belongs.


"""it has to have a list of neurons that will be "sitting" on that .
   And a neuron_count, so that the layer can know how many Neurons it's supposed to have
   and previous_layer set to None by default, this is purely for Neuron initialization."""


class Layer:
    def __init__(self, network, neuron_count: int, previous_layer=None):
        self.network = network
        self.neurons = [Neuron(self, previous_layer) for neuron in range(neuron_count)]
        # Using list comprehension here to add neurons to it according to the count.
        # Self itself here is the "layer" as it is the instance of Neuron.

        """We need to define helper methods in this Layer class as well. just like neurons,
         the layers may end up as being the "input layer" or the "output layer"
         for the input layer, all neurons have to have their output set to the passed-in values"""

        """this is layer's job - it's going to take a list of inputs,
         and instruct each of its neurons to set it's output to that.
         since this will be primarily used by the input layer only, I called it set_inputs"""

    def set_inputs(self, inputs):

        # Check the amount of inputs passed in matches number of neurons in layer

        if len(self.neurons) != len(inputs):
            raise IntegrityError("Incorrect number of inputs")  # raising the exception we defined earlier

        """Now that we know the lengths are the same we can process them
           We will use the inbuilt zip() that takes X iterables and produces
           X-length tuples where each one consists of the corresponding elements."""

        """you can zip(inputs, self.neurons) into a for loop so it'll automatically 
           unpack into those two loop vars. then, it's just a matter of using one of our already
           defined helper functions on each neuron inside the loop.
           The length check done before that is important because zip() will end on the 
           shortest iterable passed. If their lengths are the same, then it should process
           all of the neurons and inputs properly"""

        for input, neuron in zip(inputs, self.neurons):
            neuron.set_output(input)

        """so you have a neuron;
           and the value it's output should be set to.
           so, it's literally one line inside the loop
           just call that method with the value given"""

    """we've just covered the case where the layer would be used as the input layer
       now, it's time to cover the case where it'd be the output layer
       for this, you have to have a way of just getting all the outputs from the entire layer"""

    def get_outputs(self):

        # We're going through each neuron in self.neurons
        # And reading its .output into a list

        return [neuron.output for neuron in self.neurons]

    """for the backpropagation algo, as a helper function,
       this is the method where you feed it the desired targets.
       it takes the current outputs, which we can easily get from self.get_outputs(), now
       calculates the errors
       by subtracting each target value from each output
       sets those errors for each neuron with neuron.set_error() we've already defined
       and finally, calculates the sum of squares of those errors, and returns it
       just so we can pass that along and thus keep track of the teaching progress"""

    def set_errors(self, targets: List[float]):
        if len(self.neurons) != len(targets):
            raise IntegrityError("Incorrect number of outputs")
        outputs = self.get_outputs()

        # Now that we are sure that targets and self.neurons are the same length,
        # We can zip() together targets and outputs now

        errors = [(target - output) for target, output in zip(targets, outputs)]

        # Errors are the difference b/w output value and targte value obviously
        # By doing the above list comp, we get a list of errors
        # Now to set the errors to the neurons with methods we've defined already:

        for neuron, error in zip(self.neurons, errors):
            neuron.set_error(error)

        # Finally return the sum of squared values of the errors

        return sum(e ** 2 for e in errors)  # The sum iterable is being used
        # Using a list comprehension, the logic is easy

    """Another helper function, it should iterate over all neurons on the layer, and call their 
       corresponding method as well.this is just a quick way of switching the activation 
       function on the entire layer at once."""

    def set_activation_function(self, func):
        for neuron in self.neurons:
            neuron.set_activation_function(func)

    # Another helper method that is cool to define :

    def __len__(self):
        return len(self.neurons)
    # This will enable Layer class instances to support len()
    # Doing len() on a layer will return the amount of neurons inside

    def feed_forward(self):  # The layer part of feed forward's algo
        for neuron in self.neurons:
            neuron.feed_forward()  # As you can see, doing a feed forward on the entire layer;
            # Is the same as feeding forward every neuron in that layer
            # "neuron" here is each element in self.neurons and thus already derives from the class Neuron
            # So when we say feed_forward(), it derives from the definition in Neuron

    def back_propagate(self):
        for neuron in self.neurons:
            neuron.back_propagate()  # Same as for feeding forward the layer, we just have to back propagate every -
        # - neuron in that layer

    # Now continuing with the export data method

    def export_data(self) -> List[float]:
        data = []
        for neuron in self.neurons:
            data.extend(neuron.export_data())  # We can't use append or we'd get a list of lists
        return data

    # All we did there was make a long list of all the data from exporting each neuron
    # And now a method that will later on convert that long list back to a network

    def import_data(self, data_iter: Iterator[float]):
        for neuron in self.neurons:  # For each neuron in the list
            neuron.import_data(data_iter)  # Import the data back


# Now a Network class which will house everything


class Network:
    def __init__(self, layer_counts: List[int], learning_rate=0.05, momentum_mod: float = 0.05):
        # We want a list of integers as input for layer_counts

        if len(layer_counts) < 2:
            raise IntegrityError("A network cannot have less than 2 layers")
        self.layers = []  # An attribute to store the layers in

        """ a for loop, that will iterate over layer_counts
           create a layer, passing in the network (self, just like you've learned with the Layer),
           neuron count,
           and the previous layer bit"""

        layer = None

        # We have to set layer to none beforehand as for the first iteration it doesn't exist

        for count in layer_counts:
            layer = Layer(self, count, layer)
            self.layers.append(layer)

            # you can see that the "previous layer" from the previous loop will be used inside the constructor,
            # before a new Layer instance is created and overwrites the layer variable

        # The network will have two helper attributes, self.input_layer and self.output_layer

        self.input_layer = self.layers[0]  # First element
        self.output_layer = self.layers[-1]  # Last element
        self.learning_rate = learning_rate  # Attribute from which the value can be accessed later on
        self.momentum_mod = momentum_mod
        self.fitness = 0

    # And now another helper method whose implementation is similar to Layer's corresponding method:

    def set_activation_function(self, func):
        for layer in self.layers:
            layer.set_activation_function(func)

    """"The final definition for the feed forward algo pertaining to the Network as a whole:
        This is the main function so it will take an inputs parameter as well.
        This param will be a list of inputs we are passing into the network.
        Then we feed forward on all layers of the network starting from input one all the way to output one.
        Once done, you need to get the outputs from the output layer and return them.
        This feed_forward can be then used at any time later on, to simply evaluate the whole network
        feed it inputs, and get back the outputs"""

    def feed_forward(self, inputs: List[float]):
        self.input_layer.set_inputs(inputs)
        for layer in self.layers:
            layer.feed_forward()
        return self.output_layer.get_outputs()

    """on the network class, where you're currently implementing feed_forward
       you also have self.input_layer and self.output_layer
       You already have all the tools you need
       it's just a matter of calling them in a specific order
       set the inputs on the input layer
       feed forward the entire network, from the input layer to the output layer
       get the outputs from the output layer, and return them"""

    # And now to back propagate

    def back_propagate(self, targets: List[float]):
        current_error = self.output_layer.set_errors(targets)
        for layer in reversed(self.layers):
            layer.back_propagate()
        return current_error

    """if you'll take a look at the Layer class, you'll see that there's a set_errors method
       that takes a list of target values, in preparation for the back propagation to happen
       and generates the errors for the output layer, that will be propagated backwards
       so, for the network, you want to start with the same method, only that now it should take that list of targets to
       pass into that method.
       inside, you have to use that set_errors method on the output layer, passing in those targets.
       note that set_errors returns that squared error that lets you keep track of the teaching progress, so you'll have
       to store its result in some variable. then, you have to call back_propagate on every layer, but starting from 
       the output one and ending up on the input one - in the reverse order"""

    # Now a method to teach the network:

    def teach(self, dataset: list):  # We could have made the typing more accurate by making it dataset: List[Entry]
        # But to do that, we would have to define the Entry class (given below) somewhere above the network class
        # We immediately want a variable that'll hold the sum of the errors returned for each dataset entry:
        net_error = 0
        # Now we want to iterate over our dataset entries. It's also a good time to split the tuple into two
        for inputs, targets in dataset:
            self.feed_forward(inputs)  # We feed forward the entire network
            net_error += self.back_propagate(targets)  # Then back propagate it to get error.We want this value down
        return net_error

    # And now the final export method for data conversion

    def export_data(self) -> List[float]:
        data = []
        for layer in self.layers:
            data.extend(layer.export_data())
        return data

    # This is the same logic as each neuron in a layer
    # But for each layer in a network.

    # Now to import the data back, we need a method:

    def import_data(self, data: List[float]):
        with suppress(TypeError):  # Similar to try-excepting errors
            if len(data) != sum(len(n.dendrons)+1 for l in self.layers for n in l.neurons):
                raise IntegrityError("Invalid import data size")  # We have to make sure the data isn't too little or too
            # much. So we compare the length of the original list with our given data and raise an error for anomalies
        data_iter = iter(data)  # We are getting iterator from the data list, starting the process of reconversion
        for layer in self.layers:
            layer.import_data(data_iter)  # This import_data belongs to the layer class

    # This is the genetic algo logic for breeding networks together. This uses iterator and iterable concepts
    # To facilitate the crossover methods

    # Now for the methods that we'll use for the genetic algo itself, the others being helper functions
    # There will be one for the crossover and one for mutation

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

    # Now a mutation method to alter the weights during crossover genetics

    def mutate(self):
        for layer in self.layers:  # For each layer in the network
            for neuron in layer.neurons:  # For each neuron of each layer
                for dendron in neuron.dendrons:  # For each dendron connected to each neuron
                    if random.random() < 0.05:  # The mutation chance is set as a %
                        dendron.weight += dendron.weight * (0.2 * random.random() - 0.1)  # This is the actual mutation


# At this point we have finished the classes of logic for the neural network itself
# From here on we will be things like Datasets that we can feed into our neural net as well as Entries to make it simple
# Datasets generally consist of inputs and outputs. A training set would contain inputs and expected outputs


# And in doing so we have defined classes for entries. For example, defining a dataset would be like so:
# This is how we define an "Entry" which is an object of the namedtuple class that we immported:

# Entry = namedtuple("Entry", ["inputs", "outputs"])

# We will be running only example code in this file as it contains the brains of our project. Use the above line as a
# reference when calling these classes elsewhere


# Similarly this is how we define datasets (values depend on you):
# This particular set contains inputs for 2 neurons and 3 expected outputs for the output layer
# The number of values corresponds to the number of neurons in the input and output layers


"""dataset = [
    Entry([0, 0], [0, 0, 0]),
    Entry([0, 1], [1, 0, 1]),
    Entry([1, 0], [1, 0, 1]),
    Entry([1, 1], [1, 1, 0]),
]"""

# Now some example code to see this network in action :)


if __name__ == "__main__":

    # Creating a namedtuple object
    Entry = namedtuple("Entry", ["inputs", "outputs"])

    dataset = [
        Entry([0, 0], [0, 0, 0]),
        Entry([0, 1], [1, 0, 1]),
        Entry([1, 0], [1, 0, 1]),
        Entry([1, 1], [1, 1, 0]),
    ]

    # Creating a network object, a network with 2 input neurons, a hidden layer with 3 neurons and an output layer of 3


    net = Network([2, 3, 3])

    precision = 0.1
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

    """net = Network([2, 3, 3])
    dataset1 = net.export_data()
    net.import_data(dataset1)
    dataset2 = net.export_data()
    assert dataset1 == dataset2
    exit()"""

    # The above is just some code we used to test whether the import/export logic works and doesn't cause issues
