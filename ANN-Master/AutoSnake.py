# Here we will inherit from our classes over in the main file and snake files
# Then we will use the genetic algo and make the snake try to learn how to play itself

from NeuralNet import Network
from pyglet_snake import Game
from functools import partial
import random
from time import sleep
from typing import Union, Generator, List
import multiprocessing as mp

# We need an infinite generator to be able to import the values from the import function differently so we can play with
# weights.


def random_gen() -> Union[Generator[float, None, None], List[float]]:
    while True:
        yield 2 * random.random() - 1


# Will be using this to clamp down some of the input sizes later
def clamp(val):
    if val > 100:
        return 100
    elif val < -100:
        return -100
    return val


# We need a couple of helper functions to handle the "breeding" of the generations in our population
# this one actually runs the game

def controller(network, game):
    # Getting the head positions:
    hx, hy = game.head.position
    # Getting the apple positions:
    ax, ay = game.apple.position
    # Positions for the walls:
    wx, wy = game.window.get_size()
    # Positions for the current directions:
    current_dirs = [0, 0, 0, 0]  # we'll extend this list into the inputs
    current_dirs[game.direction] = 1  # Assign 1 to the index of current direction

    # Forbidden directions
    forbidden_dirs = [0, 0, 0, 0]
    for seg in game.snake:
        if hx == seg.x:
            if seg.y == hy + 20:
                forbidden_dirs[1] = 1
            elif seg.y == hy - 20:
                forbidden_dirs[3] = 1
        elif hy == seg.y:
            if seg.x == hx + 20:
                forbidden_dirs[0] = 1
            elif seg.x == hx - 20:
                forbidden_dirs[2] = 1
    if hx + 20 > wx:
        forbidden_dirs[0] = 1
    elif hx < 20:
        forbidden_dirs[2] = 1
    if hy < 20:
        forbidden_dirs[3] = 1
    elif hy + 20 > wy:
        forbidden_dirs[1] = 1

    # lower_wall = hy
    # left_wall = hx
    # right_wall = wx - hx
    # upper_wall = wy - hy
    # body_x = min([seg.x - hx for seg in game.snake]) -> don't use lol
    # body_y = min([seg.y - hy for seg in game.snake]) -> ditto

    inputs = [clamp(ax - hx)/100, clamp(ay - hy)/100]
    inputs.extend(current_dirs)
    inputs.extend(forbidden_dirs)

    # That makes 10 inputs

    # evaluate the network
    outputs = network.feed_forward(inputs)
    # for o in outputs:
    #   print(o)
    # print("\n")

    # Outputs are in a list
    # use the outputs to change the direction
    game.direction = outputs.index(max(outputs))  # Use common sense


def crossover_networks(nets):
    for n1, n2 in zip(nets[:-1:2], nets[1::2]):  # skipping by twos (for single point crossover)
        n3, n4 = n1.crossover(n2)
        nets.extend((n3, n4))


def mutate_networks(nets):  # Just calling the mutate function for every net
    for net in nets:
        net.mutate()


def eval_net(net):  # To evaluate and calculate fitness etc. per each network
    # reset the game to the starting position
    game.reset()
    # attach the controller
    game.external = partial(controller, net)
    # run the game and calculate the fitness
    game.run()
    # Fitness calc:
    if game.score == 0 or game.exit_code == 3:
        return 0
    else:
        return game.score * 1000 + (500 - game.steps)


def evaluate_networks(pool, nets):  # This will be what helps run the actual game
    if switch:
        for net, fitness in zip(nets, pool.imap(eval_net, nets)):
            net.fitness = fitness
    else:
        for net in nets:
            net.fitness = eval_net(net)


"""Note for above ^ :
when you attach an external controller
it gets called with exactly one parameter - the game instance itself
but, for your neural network switching, you need to be able to pass in the network too
note - you're attaching the controller itself
an entire function
like:
game.external = controller
you don't call it - the game calls it
internally, on every update loop
you can't really pass in the network to that
game.external = controller(net) won't work, because that's a call
game.external = controller(net, game) won't work either, because that's a call too.
and you can't pass in the controller alone because it takes two parameters
and the game will internally call it with only one, the game instance itself
so we need a partial object here
the syntax is simple
partial(func, parameters)
Parameters can come as they normally do, and you don't have to fill in all of them
the signature has network as the first parameter for a reason
game.external = partial(controller, net)
This: partial(controller, net)
will create a function
that has the first parameter already pre-filled
the network
the only other parameter left - the game instance
will be passed internally by the game itself, once it calls the partial object
"""

# Note, you evaluate first and then crossover, don't pay attention to the order in my execution

# Now the main code:
# Note, the things inside the main clause are very specifically put here keeping multiprocessing in mind
# But it will work even with the multiprocessing turned off

#################################
# Main multiprocessing switch ###
#################################
switch = True  # ################
#################################

# Now to initialize the snake game. We aren't straight up initializing the class because we have to do it for each
# Worker in each process
if switch:
    game = None
else:
    game = Game()
    game.fps = 6000

# Similarly for barriers in multiprocessing
barrier = None


# This will be the initializer for the processes
def initializer(bar):
    global game
    # init the snake game
    game = Game()
    # run it fast
    game.fps = 600
    # setup the finalizer barrier
    global barrier
    barrier = bar


# This is the finalizer required to close each process once they're freed from the barrier
def finalizer(wait: bool):
    game.exit()
    if wait:
        barrier.wait()


if __name__ == "__main__":
    if switch:
        # getting cpu var for the pc this runs on
        cpu_count = mp.cpu_count()
        # barrier obj
        barrier = mp.Barrier(cpu_count)
        # pool obj
        pool = mp.Pool(cpu_count, initializer, (barrier,))
    else:
        pool = None
    # First we need to define our population of networks, let's go with 200 per generation:
    networks = [Network([10, 8, 6, 4]) for _ in range(200)]
    for net in networks:
        net.import_data(random_gen())  # We suppress this typing error later on
        # In this way we don't actually have to mess with the weights definition in the NeuralNet file
    generation = 0
    best_network = None
    max_generation = 200
    while True:
        generation += 1
        # evaluate networks for fitness
        evaluate_networks(pool, networks)
        # sort networks with fittest at the very top
        networks.sort(key=lambda n: n.fitness, reverse=True)  # Descending list of network fitnesses
        # save the best network (optional)
        if not best_network or best_network.fitness < networks[0].fitness:
            best_network = networks[0]
        print(f"Generation no: {generation}; Best Fitness: {best_network.fitness}")
        if generation >= max_generation:
            break
        # discard lower 50% of the population
        networks = networks[:len(networks) // 2]
        # crossover the networks to get back to 100% population
        crossover_networks(networks)
        # mutate the networks
        mutate_networks(networks)

    # This is the switch that turns on and off multiprocessing, it is True by default, set it to false for slower
    # machines or if you want to keep better track

    if switch:
        # finalize all worker processes
        pool.map(finalizer, (True for _ in range(cpu_count)))
        pool.close()
        pool.join()  # closing the pool and joining all the processes back to the main one
    else:
        game.exit()

    # And now to watch the final generation in action:
    # create an entirely new game instance
    game = Game()
    # attach the controller
    game.external = partial(controller, best_network)
    game.fps = 30
    while not game.window.has_exit:
        # reset the game
        game.reset()
        # run the game
        game.run()
        print(f"Score: {game.score}")
        sleep(1)
