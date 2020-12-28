import tensorflow.keras
import pygad.kerasga
import numpy
import pygad
from tensorflow.keras.constraints import min_max_norm
from tensorflow.keras import initializers
import random
import bitstring

def fitness(solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model
    
    weight_list = []
    for i in range(0, len(solution), 15):
        binary_as_list = solution[i : i+15]
        a = bitstring.BitArray(bin=("".join(str(i) for i in binary_as_list)))
        list_to_decimal = a.int
        weight_list.append(list_to_decimal)
    # print(weight_list)

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=weight_list)

    model.set_weights(weights=model_weights_matrix)

    predictions = model.predict(data_inputs)

    mae = tensorflow.keras.losses.MeanAbsoluteError()
    abs_error = mae(data_outputs, predictions).numpy() + 0.00000001
    solution_fitness = 1.0 / abs_error

    return solution_fitness

def gen_initial_population():
    population = []
    for _ in range(100):
        neural_ind = []
        for _ in range(25):
            weight = random.randint(-10,10)
            to_bin = lambda x, count=8: "".join(map(lambda y:str((x>>y)&1), range(count-1, -1, -1)))
            neural_ind.extend(to_bin(weight, 15))
        population.append(neural_ind)
    return population

def callback_generation(ga_instance):
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))


# Neural network has one hidden layer with six neurons.
input_layer = tensorflow.keras.layers.Input(2)
dense_layer1 = tensorflow.keras.layers.Dense(
    6, activation="sigmoid")(input_layer)
output_layer = tensorflow.keras.layers.Dense(
    1, activation="linear")(dense_layer1)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

# print(model.get_weights())

# use the model, set population size to 100
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

data_inputs = []
data_outputs = []

# read training data from the file
file_t = open("train.dat", "r")
lines_t = file_t.readlines()
for line in lines_t:
    x1, x2, y = line.split()
    data_inputs.append([float(x1), float(x2)])
    data_outputs.append([float(y)])

num_generations = 100
num_parents_mating = 2

initial_pop = gen_initial_population()

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness,
                       initial_population=initial_pop,
                       on_generation=callback_generation,
                       random_mutation_min_val=0,
                       random_mutation_max_val=2,
                       mutation_by_replacement=True,
                       gene_type=int,
                       mutation_percent_genes=10
                       )
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(
    title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                              weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(data_inputs)
print("Predictions : \n", predictions)

mae = tensorflow.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print("Absolute Error : ", abs_error)
