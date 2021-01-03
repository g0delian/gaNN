import random
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator, base, tools, algorithms
import numpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

popSize = 28  # Population size
dimension = 3  # Number of decision variable x
numOfBits = 10  # Number of bits in the chromosomes
iterations = 100  # Number of generations to be run
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

toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)
toolbox.register("select", tools.selNSGA2)


def efficient_sort(pop):
    # sort according to f1
    sorted_data = sorted(pop, key=lambda x: x.fitness.values[0])
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
            for gen_idx in range(len(pareto_idx[front_idx]))[::-1]:
                gene = pareto_idx[front_idx][gen_idx]
                # print(gen_idx)
                if not (f1 < gene.fitness.values[0] or f2 < gene.fitness.values[1]):
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
    # Encode the decision variables using Gray coding, each using 10
    # its. Set the population size to 28 and randomly generate an
    # initial population.
    pop = toolbox.population(n=popSize)
    # for graphics
    x1_l = []
    x2_l = []
    x3_l = []
    f1_values = []
    f2_values = []
    # evaluate
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        # print(ind)
        x1, x2, x3 = separatevariables(ind)
        x1_l.append(x1)
        x2_l.append(x2)
        x3_l.append(x3)
        ind.fitness.values = fit
        f1_values.append(fit[0])
        f2_values.append(fit[1])

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"type": "table"}],
               [{"type": "table"}]]
    )

    # efficient sort and front values
    f1_value = []
    f2_value = []
    front_value = []
    sort_pop = efficient_sort(pop)
    for x in range(len(sort_pop)):
        for j in sort_pop[x]:
            f1_value.append(j.fitness.values[0])
            f2_value.append(j.fitness.values[1])
            front_value.append(x)

    # create a graph
    fig.add_trace(
        go.Table(
            header=dict(
                values=["x1", "x2", "x3", "f1", "f2"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[x1_l, x2_l, x3_l, f1_values, f2_values],
                align="left")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["f1", "f2", "front"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[f1_value, f2_value, front_value],
                align="left")
        ),
        row=2, col=1
    )

    # max objective values
    worst_f1 = max(pop, key=lambda x: x.fitness.values[0])
    worst_f2 = max(pop, key=lambda x: x.fitness.values[1])
    print("Max f1: ", worst_f1.fitness.values[0])
    print("Max f2", worst_f2.fitness.values[1])

    # no actual selection, crowding values only.
    pop = toolbox.select(pop, len(pop))

    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Question 1: Answers 1, 2, 3 <br> <br> They are scrolable!",
    )
    fig.show()

    # turn to true for acquire graphs every run
    # graphs need to be closed to proceed
    graph_every_run = False
    hyper_volume_l = []
    for gen in range(1, NGEN):
        # Apply the binary tournament selection to select parent
        # individuals for reproduction.
        parents = tools.selTournamentDCD(pop, len(pop))

        offspring = [toolbox.clone(ind) for ind in parents]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(ind1, ind2, crossProb)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # evaluate offsprings
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Then select 25 individuals from the combined
        # population using the crowded non-dominated sorting method.
        pop = toolbox.select(parents + offspring, popSize)

        if graph_every_run:
            # Plot individuals together with the parents in the objective space.
            x_parents = [x.fitness.values[0] for x in parents]
            y_parents = [x.fitness.values[1] for x in parents]

            fig, ax = plt.subplots()

            plt.scatter(x_parents, y_parents, c='blue', label="parents")

            x_siblings = [x.fitness.values[0] for x in offspring]
            y_siblings = [x.fitness.values[1] for x in offspring]

            ax.scatter(x_siblings, y_siblings, c='red', label="siblings")

            plt.title('Generation {} Parents and Siblings'.format(gen))
            plt.xlabel('f1')
            plt.ylabel('f2')
            #plt.savefig('parents_siblings_objective.png')
            ax.legend()
            plt.show()

            # Plot the combined population (50 individuals) in the objective
            # space. Highlight the 25 selected solutions.
            x_pop = [x.fitness.values[0] for x in parents + offspring]
            y_pop = [x.fitness.values[1] for x in parents + offspring]

            fig, ax = plt.subplots()

            plt.scatter(x_pop, y_pop, c='blue', label="population")

            x_selected = [x.fitness.values[0] for x in pop]
            y_selected = [x.fitness.values[1] for x in pop]

            ax.scatter(x_selected, y_selected, c='red', label="selected")

            plt.title('Generation {} Population Selection'.format(gen))
            plt.xlabel('f1')
            plt.ylabel('f2')
            #plt.savefig('population_selection_objective.png')
            ax.legend()
            plt.show()

        # calculate according to max(f1) and max(f2)

        print("Final population hypervolume is %f" %
          hypervolume(pop, [worst_f1.fitness.values[0], worst_f2.fitness.values[0]]))
        hyper_volume_l.append(hypervolume(pop, [worst_f1.fitness.values[0], worst_f2.fitness.values[0]]))

    plt.figure()
    plt.plot(range(len(hyper_volume_l)), hyper_volume_l, linewidth=3)
    plt.title("Hypervolume Over Generations")
    plt.ylabel('Hypervolume')
    plt.xlabel('Generation')
    plt.show()

if __name__ == "__main__":
    main()
