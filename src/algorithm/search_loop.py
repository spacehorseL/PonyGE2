from multiprocessing.pool import ThreadPool
from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from stats.stats import stats, get_stats
from utilities.stats import trackers
from utilities.stats.logger import Logger
from operators.initialisation import initialisation
from utilities.algorithm.initialise_run import pool_init

def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = ThreadPool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    Logger.log("Generation 0 starts. Initializing...")
    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])

    # Evaluate initial population
    individuals = evaluate_fitness(individuals)

    # Generate statistics for run so far
    get_stats(individuals)

    # Cleanup after evaluation
    if 'cleanup' in dir(params['FITNESS_FUNCTION']):
        params['FITNESS_FUNCTION'].cleanup(individuals)

    # Traditional GE
    for generation in range(1, (params['GENERATIONS']+1)):
        stats['gen'] = generation
        params['CURRENT_EVALUATION'] = 0
        params['CURRENT_GENERATION'] = generation
        Logger.log("Generation {0} starts...".format(generation))
        # New generation
        individuals = params['STEP'](individuals)
        # Cleanup after evaluation
        if 'cleanup' in dir(params['FITNESS_FUNCTION']):
            params['FITNESS_FUNCTION'].cleanup(individuals)

    Logger.close_all()
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals


def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    individuals = trackers.state_individuals

    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation

        # New generation
        individuals = params['STEP'](individuals)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals
