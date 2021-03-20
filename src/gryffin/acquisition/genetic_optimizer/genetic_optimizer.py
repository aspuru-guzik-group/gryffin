#!/usr/bin/env python 

__author__ = 'Florian Hase'


import numpy as np
from gryffin.utilities import Logger, GryffinUnknownSettingsError
from gryffin.observation_processor import param_vector_to_dict
from deap import base, creator, tools, algorithms


class GeneticOptimizer(Logger):

    def __init__(self, config, known_constraints=None):
        self.config = config
        self.known_constraints = known_constraints
        Logger.__init__(self, 'GeneticOptimizer', verbosity=self.config.get('verbosity'))

    def acquisition(self, x):
        return self._acquisition(x),

    def set_func(self, acquisition, ignores=None):
        self._acquisition = acquisition

        if any(ignores) is True:
            raise NotImplementedError('GeneticOptimizer with process constraints has not been implemented yet. '
                                      'Please choose "adam" as the "acquisition_optimizer".')

    def optimize(self, samples, max_iter=10, verbose=True):

        # crossover and mutation probabilites
        CXPB = 0.5
        MUTPB = 0.3

        if self.acquisition is None:
            self.log('cannot optimize without a function being defined', 'ERROR')
            return None

        # setup GA with DEAP
        creator.create("FitnessMin", base.Fitness, weights=[-1.0])  # we minimize the acquisition
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # ------------
        # make toolbox
        # ------------
        toolbox = base.Toolbox()
        toolbox.register("population", param_vectors_to_deap_population)
        toolbox.register("evaluate", self.acquisition)
        # use custom mutations for continuous, discrete, and categorical variables
        toolbox.register("mutate", self._custom_mutation, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # mating type depends on how many genes we have
        if np.shape(samples)[1] == 1:
            toolbox.register("mate", cxDummy)  # i.e. no crossover
        elif np.shape(samples)[1] == 2:
            toolbox.register("mate", tools.cxUniform, indpb=0.5)  # uniform crossover
        else:
            toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover

        # ---------------------
        # Initialise population
        # ---------------------
        population = toolbox.population(samples)

        # Evaluate pop fitnesses
        fitnesses = list(map(toolbox.evaluate, np.array(population)))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # create hall of fame
        num_elites = int(round(0.05 * len(population), 0))  # 5% of elite individuals
        halloffame = tools.HallOfFame(num_elites)  # hall of fame with top individuals
        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0

        # register some statistics and create logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose is True:
            split_stream = logbook.stream.split('\n')
            self.log(split_stream[0], 'INFO')
            self.log(split_stream[1], 'INFO')

        # ------------------------------
        # Begin the generational process
        # ------------------------------
        for gen in range(1, max_iter + 1):

            # Select the next generation individuals (allow for elitism)
            offspring = toolbox.select(population, len(population) - hof_size)

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, np.array(invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # add the best back to population
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose is True:
                self.log(logbook.stream, 'INFO')

            # convergence criterion, if the population has very similar fitness, stop
            if record['std'] < 0.01:  # i.e. ~1% of acquisition codomain
                break

        # DEAP cleanup
        del creator.FitnessMin
        del creator.Individual

        return np.array(population)

    def _unconstrained_evolution(self):
        pass

    def _constrained_evolution(self):
        pass

    def _custom_mutation(self, individual, indpb=0.2, continuous_scale=0.1, discrete_scale=0.1):
        """Custom mutation that can handled continuous, discrete, and categorical variables.

        Parameters
        ----------
        individual :
        indpb : float
            Independent probability for each attribute to be mutated.
        continuous_scale : float
            Scale for normally-distributed perturbation of continuous values.
        discrete_scale : float
            Scale for normally-distributed perturbation of discrete values.
        """

        assert len(individual) == len(self.config.param_types)

        for i, param in enumerate(self.config.parameters):
            param_type = param['type']

            # determine whether we are performing a mutation
            if np.random.random() < indpb:

                if param_type == "continuous":
                    # Gaussian perturbation with scale being 0.1 of domain range
                    bound_low = self.config.feature_lowers[i]
                    bound_high = self.config.feature_uppers[i]
                    scale = (bound_high - bound_low) * continuous_scale
                    individual[i] += np.random.normal(loc=0.0, scale=scale)
                    individual[i] = _project_bounds(individual[i], bound_low, bound_high)
                elif param_type == "discrete":
                    # add/substract an integer by rounding Gaussian perturbation
                    # scale is 0.1 of domain range
                    bound_low = self.config.feature_lowers[i]
                    bound_high = self.config.feature_uppers[i]
                    # if we have very few discrete variables, just move +/- 1
                    if bound_high - bound_low < 10:
                        delta = np.random.choice([-1, 1])
                        individual[i] += delta
                    else:
                        scale = (bound_high - bound_low) * discrete_scale
                        delta = np.random.normal(loc=0.0, scale=scale)
                        individual[i] += np.round(delta, decimals=0)
                    individual[i] = _project_bounds(individual[i], bound_low, bound_high)
                elif param_type == "categorical":
                    # resample a random category
                    num_options = float(self.config.feature_sizes[i])  # float so that np.arange returns doubles
                    individual[i] = np.random.choice(list(np.arange(num_options)))
                else:
                    raise ValueError()
            else:
                continue

        return individual,


def cxDummy(ind1, ind2):
    """Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
    crossover would not change the population.
    """
    return ind1, ind2


def _project_bounds(x, x_low, x_high):
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


def param_vectors_to_deap_population(param_vectors):
    population = []
    for param_vector in param_vectors:
        ind = creator.Individual(param_vector)
        population.append(ind)
    return population