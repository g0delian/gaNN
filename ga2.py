import random
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator, base, tools, algorithms
import numpy
from tabulate import tabulate

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

popSize = 28  # Population size
dimension = 3  # Number of decision variable x
numOfBits = 10  # Number of bits in the chromosomes
iterations = 100  # Number of generations to be run
crossPoints = 2  # variable not used. instead tools.cxTwoPoint
crossProb = 0.9
flipProb = 1. / (dimension * numOfBits)  # bit mutate prob
mutateprob = .1  # mutation prob
maxnum = 2**numOfBits

BOUND_LOW, BOUND_UP = -4, 4

NGEN = 30

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, numOfBits*dimension)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)
toolbox.register("select", tools.selNSGA2)


def efficient_sort(pop):
    # sort according to f1
    sorted_data = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
    # print(sorted_data)
    pareto_idx = [[]]  # list of lists for keeping fronts
    pareto_idx[0].append(sorted_data[0])  # add the first data
    
    for check_gen in range(1, len(sorted_data)):
        f1 = sorted_data[check_gen].fitness.values[0]
        f2 = sorted_data[check_gen].fitness.values[1]
        no_dominate = True
        for front_idx in range(len(pareto_idx)):
            dominates = True
            front_count = len(pareto_idx)
            for gen_idx in range(len(pareto_idx[front_idx])):
                gene = pareto_idx[front_idx][gen_idx]
                # print(gen_idx)
                if not (f1 > gene.fitness.values[0] or f2 > gene.fitness.values[1]):
                    #print(gene.fitness.values[0], f1)
                    #print(gene.fitness.values[1], f2)
                    dominates = False
                    # print(dominates)
                    break
            if dominates:
                pareto_idx[front_idx].append(sorted_data[check_gen])
                no_dominate = False
                break
        if no_dominate:
            pareto_idx.append([])
            pareto_idx[front_count].append(sorted_data[check_gen])

    return pareto_idx


def eval_sphere(individual):
    sep = separatevariables(individual)
    f1 = (((sep[0]/2.0)**2 + (sep[1]/4.0)**2 + (sep[2])**2))/3.0
    f2 = (((sep[0]/2.0-1.0)**2 + (sep[1]/4.0-1.0)**2 + (sep[2]-1.0)**2)) / 3.0
    return f1, f2


toolbox.register("evaluate", eval_sphere)


def chrom2real(c):
    indasstring = ''.join(map(str, c))
    degray = gray_to_bin(indasstring)
    numasint = int(degray, 2)  # convert to int from base 2 list
    numinrange = -5+10*numasint/maxnum
    return numinrange


# input: concatenated list of binary variables
# output: tuple of real numbers representing those variables

def separatevariables(v):
    return chrom2real(v[0:numOfBits]), chrom2real(v[numOfBits:20]), chrom2real(v[numOfBits*2:])


def main():
    # random.seed(64)
    # print("INITIAL POPULATION")
    pop = toolbox.population(n=popSize)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        # print(ind)
        x1, x2, x3 = separatevariables(ind)
        ind.fitness.values = fit
        # print(tabulate([[x1, x2, x3, fit[0], fit[1]], ], headers=[
        # 'x1', 'x2', 'x3', 'f1 fitness', 'f2 fitness']))

    sort_pop = efficient_sort(pop)
    for x in range(len(sort_pop)):
        print(x, sort_pop[x])

    # no actual selection, crowding values only.
    pop = toolbox.select(pop, len(pop))

    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        # selTournamentDCD means Tournament selection based on dominance (D)
        # followed by crowding distance (CD). This selection requires the
        # individuals to have a crowding_dist attribute
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # make pairs of all (even,odd) in offspring
            if random.random() <= crossProb:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, popSize)

    print("Final population hypervolume is %f" %
          hypervolume(pop, [11.0, 11.0]))


if __name__ == "__main__":
    main()
