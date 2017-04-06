import math
import random

import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", set, fitness=creator.FitnessMax)

# list of available times from people
# tuples in the form: (ID, (DAY, TIME), PREFERENCE)
# preference is in form high value = higher preference
available_times = []

# generate dummy data
for i in range(10):
    for j in range(random.randint(0, 5)):
        day = random.choice(['M', 'TU', 'W', 'TH', 'F'])
        slot = random.choice(['11-1', '1-3', '3-5'])
        pref = random.randint(1, 9)
        available_times.append((i, (day, slot), pref))  # ID, (Day, Slot), Preference

IND_INIT_SIZE = 10
MAX_SLOTS = 3  # max number of slots a person can fill

toolbox = base.Toolbox()
toolbox.register("attribute", random.randrange, len(available_times))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalSchedule(individual):
    value = 0.0
    pids = []
    day_slots = []

    for i in individual:
        item = available_times[i]
        pid = item[0]
        ds = item[1]
        pref = item[2]

        if pids.count(pid) < MAX_SLOTS:
            value += (MAX_SLOTS * 10) * math.exp(-2 * pids.count(pid)) * 10

        if ds not in day_slots:
            value += 1000
        elif day_slots.count(ds) < 2:
            value += 250

        value += pref * 5

        pids.append(pid)
        day_slots.append(ds)

    return value,


def cxSet(ind1, ind2):
    temp = set(ind1)
    ind1 &= ind2
    ind2 ^= temp

    return ind1, ind2


def mutSet(individual):
    if random.random() < 0.5 and len(individual) > 0:
        individual.remove(random.choice(sorted(tuple(individual))))
    elif len(individual) < 15:
        individual.add(random.randrange(len(available_times)))

    return individual,


toolbox.register("evaluate", evalSchedule)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selBest)


def main():
    NGEN = 100  # number of generations
    MU = 50
    LAMBDA = 100
    CXPB = 0.7  # crossover
    MUTPB = 0.2  # mutation

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)

    # print(pop)
    # print(stats)
    # print(hof)

    return hof


if __name__ == '__main__':
    best = main()

    print("RESULTS")
    for i in best[0]:
        print(available_times[i])
