import tensorflow.keras
import pygad.kerasga
import numpy
import pygad
from tensorflow.keras.constraints import min_max_norm
from tensorflow.keras import initializers
import random
import bitstring
import matplotlib.pyplot as plt

x_min = -10
x_max = 10
num_bytes = 15


def fitness(solution, sol_idx):
    global training_inputs, training_outputs, keras_ga, model, testing

    weight_list = []
    for i in range(0, len(solution), 15):
        binary_as_list = solution[i: i+15]
        d_value = int(''.join(map(str, binary_as_list)), 2)
        x_actual = x_min + ((x_max - x_min) / (2**num_bytes - 1)) * d_value
        weight_list.append(x_actual)

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=weight_list)

    model.set_weights(weights=model_weights_matrix)

    if testing:
        predictions = model.predict(test_inputs)
        mae = tensorflow.keras.losses.MeanSquaredError()
        abs_error = mae(test_outputs, predictions).numpy() + 0.00000001
    else:
        predictions = model.predict(training_inputs)
        mae = tensorflow.keras.losses.MeanSquaredError()
        abs_error = mae(training_outputs, predictions).numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error

    return solution_fitness


def lt_learning(solution_idx):
    global model
    solution = ga_instance.population[solution_idx]
    weight_list = []
    for i in range(0, len(solution), 15):
        binary_as_list = solution[i: i+15]
        d_value = int(''.join(map(str, binary_as_list)), 2)
        x_actual = x_min + ((x_max - x_min) / (2**num_bytes - 1)) * d_value
        weight_list.append(x_actual)

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=weight_list)

    model.set_weights(weights=model_weights_matrix)
    model.compile(optimizer="rmsprop", loss="mse")
    model.fit(training_inputs, training_outputs,
              epochs=3)  # 30 iterations, 3 epoch
    predictions = model.predict(training_inputs)
    local_search = pygad.kerasga.model_weights_as_vector(model)
    mae = tensorflow.keras.losses.MeanSquaredError()
    abs_error = mae(training_outputs, predictions).numpy() + 0.00000001
    solution_fitness = 1.0 / abs_error

    # print(local_search)
    lt_vector = []

    for x_acc in local_search:
        d_value = (x_acc - x_min) / ((x_max - x_min) / (2**num_bytes - 1))
        b_value = "{0:015b}".format(round(d_value))
        lt_vector.extend(b_value)

    return lt_vector, solution_fitness


def callback_generation(ga_instance):
    global training_loss
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))
    print("Loss    = {loss_value}".format(
        loss_value=1.0 / ga_instance.best_solution()[1]))
    training_loss.append(1.0 / ga_instance.best_solution()[1])


def callback_fitness(ga_instance, population_fitness):
    # skip initial pop, start from gen 1
    global best_individuals
    global initial_population
    if initial_population:
        initial_population = False
        return
    best_solt_idx = ga_instance.best_solution()[2]
    lt_vector, lt_value = lt_learning(best_solt_idx)
    # apply lamarcikian evolution (write back to genotype)
    if evolution_type == "lamarck":
        ga_instance.population[best_solt_idx] = lt_vector
        population_fitness[best_solt_idx] = lt_value
    if evolution_type == "baldwin":
        population_fitness[best_solt_idx] = lt_value

    best_individuals.append(ga_instance.population[best_solt_idx])


# Neural network has one hidden layer with six neurons.
input_layer = tensorflow.keras.layers.Input(2)
dense_layer1 = tensorflow.keras.layers.Dense(
    6, activation="sigmoid")(input_layer)
output_layer = tensorflow.keras.layers.Dense(
    1, activation="linear")(dense_layer1)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

training_inputs = []
training_outputs = []
test_inputs = []
test_outputs = []
training_loss = []

# read training data from the file
file_t = open("train.dat", "r")
lines_t = file_t.readlines()
for line in lines_t:
    x1, x2, y = line.split()
    training_inputs.append([float(x1), float(x2)])
    training_outputs.append([float(y)])

file_t = open("test.dat", "r")
lines_t = file_t.readlines()
for line in lines_t:
    x1, x2, y = line.split()
    test_inputs.append([float(x1), float(x2)])
    test_outputs.append([float(y)])

num_generations = 5
num_parents_mating = 2
evolution_type = "lamarck"  # {lamarck, baldwin}
initial_population = True
testing = False
best_individuals = []

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness,
                       sol_per_pop=100,
                       num_genes=375,
                       init_range_low=0,
                       init_range_high=2,
                       on_generation=callback_generation,
                       on_fitness=callback_fitness,
                       random_mutation_min_val=0,
                       random_mutation_max_val=2,
                       mutation_by_replacement=True,
                       gene_type=int,
                       mutation_percent_genes=5
                       )
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(
    title="Generation vs. Fitness", linewidth=4)

# losses of best individuals in every generation
plt.figure()
plt.plot(range(len(training_loss)), training_loss, linewidth=3)
plt.title("TRAINING Generation vs Loss")
plt.ylabel('Loss')
plt.xlabel('Generation')
plt.show()

# best individual in each generation
testing = True
test_loss = []
for individual in best_individuals:
    loss = 1.0 / fitness(individual, 0)
    test_loss.append(loss)

print(test_loss)

plt.figure()
plt.plot(range(len(test_loss)), test_loss, linewidth=3)
plt.title("TEST Generation vs Loss")
plt.ylabel('Loss')
plt.xlabel('Generation')
plt.show()
