import Reporter
import random
import numpy as np

import matplotlib.pyplot as plt

distance_matrix = None

# config
population_size = 100  # lambda
offspring_size = 50  # mu
k_tournament = 5  # K-Tournament Selection
stop_crit_perc = 0.001  # percentage verschil waarbij we stoppen
stop_crit_num = 6 # aantal keer percentage verschil waarbij we stoppen


# Modify the class name to match your student number.
class r0761312:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        global distance_matrix
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        ts = TravelingSalesman(population_size)
        # plt.axis([0, num_iterations, 30000, 100000])
        no_improvement_counter = 0
        mean_objective_old = 0
        iteration = 0

        while no_improvement_counter < stop_crit_num:  # yourConvergenceTestsHere
            # Your code here.
            ts.variate()
            ts.eliminate()

            if ts.mean_objective()*(1 + stop_crit_perc) > mean_objective_old:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0
            mean_objective_old = ts.mean_objective()
            iteration += 1

            print(iteration, ts.best_objective(), ts.population.size)
            # plt.scatter(iteration, ts.mean_objective(), color='red')
            # plt.scatter(iteration, ts.best_objective(), color='blue')
            # plt.pause(0.000001)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            time_left = self.reporter.report(ts.mean_objective(), ts.best_objective(), ts.best_solution())

            if time_left < 0:
                print('tijd is om')
                break

        # plt.show()
        # Your code here.
        return 0


class TravelingSalesman:

    # initialisation
    def __init__(self, population_size):
        self.population = np.array([])
        for i in range(0, population_size):
            self.population = np.append(self.population, Path(len(distance_matrix)))
        #print(self.select())

    def k_tournament(self):
        p = random.choices(self.population, k=k_tournament)
        p = sorted(p, key=lambda path: path.fitness, reverse=True)
        return p[0]

    def k_tournament_inverse(self):
        p = random.choices(self.population, k=k_tournament)
        p = sorted(p, key=lambda path: path.fitness, reverse=False)
        return p[0]

    def select(self):
        new_population = np.array([])
        population = self.population.copy()
        for i in range(0, offspring_size):
            best = self.k_tournament()
            new_population = np.append(new_population, best)
            # Optimalisatie: verwijder alleen het 1e voorkomen, dus niet door heel de lijst gaan met np.where
            population = np.delete(population, np.where(population == best))
        return new_population

    # select
    # crossover
    # mutate (met bepaalde kans)
    def variate(self):
        selected = self.select()
        for i in range(0, len(selected), 2):
            self.population = np.append(self.population, selected[i].crossover(selected[i + 1]))
        self.mutate()

    def mutate(self):
        sum = self.sum_objective()
        for solution in self.population:
            path = solution.path
            p = solution.fitness/sum
            if random.uniform(0, 1) < p:
                idx = range(len(path))
                i1, i2 = random.sample(idx[1:], 2)
                path[i1], path[i2] = path[i2], path[i1]

    def eliminate(self):
        new_population = np.array([])
        for i in range(0, population_size):
            best = self.k_tournament_inverse()
            new_population = np.append(new_population, best)
            # Optimalisatie: verwijder alleen het 1e voorkomen, dus niet door heel de lijst gaan met np.where
            self.population = np.delete(self.population, np.where(self.population == best))
        self.population = new_population

    def eliminate2(self): # beter slechtste weggooien dan beste bijhouden
        for i in range(0, int(offspring_size/2)):
            worst = self.k_tournament()
            # Optimalisatie: verwijder alleen het 1e voorkomen, dus niet door heel de lijst gaan met np.where
            self.population = np.delete(self.population, np.where(self.population == worst))

    def mean_objective(self):
        sum = 0
        for path in self.population:
            sum += path.fitness
        return sum / self.population.size

    def best_objective(self):
        best = np.Inf
        for path in self.population:
            if path.fitness < best:
                best = path.fitness
        return best

    def sum_objective(self):
        sum = 0
        for path in self.population:
            sum += path.fitness
        return sum

    def best_solution(self):
        best = np.Inf
        best_path = None
        for path in self.population:
            if path.fitness < best:
                best = path.fitness
                best_path = path
        return best_path.path


class Path:
    # Representation:
    #     Cycle with first position always 0
    def __init__(self, num_cities, path=None):
        if path is None:
            path = np.random.permutation(num_cities)
            path = path[path != 0]
            self.path = np.insert(path, 0, 0)
            self.fitness = self.calculate_fitness()
        else:
            assert path[0] == 0
            assert num_cities == path.size
            self.path = path
            self.fitness = self.calculate_fitness()

    def __repr__(self):
        return 'Path: {}'.format(self.path)

    def __le__(self, other):
        return self.fitness < other.fitness

    def set_path(self, path):
        self.path = path

    def cycle_notation(self):
        return self.path.copy()

    def adjacency_notation(self):
        length = len(self.path)
        adjacency = np.zeros(length, dtype=int)
        for i in range(length):
            j = np.where(self.path == i)[0][0]
            if j == length - 1:
                adjacency[i] = 0
            else:
                adjacency[i] = self.path[j + 1]
        return adjacency

    # input is gewoon een np array in adjacency notation
    def adjacency_to_cycle(self, adj):
        length = len(self.path)
        cycle = np.zeros(length, dtype=int)
        cycle[0] = 0
        for i in range(1, length):
            cycle[i] = adj[cycle[i - 1]]
        return cycle

    def calculate_fitness(self):
        fitness = 0
        for i in range(len(self.path) - 1):
            fitness += distance_matrix[self.path[i], self.path[i + 1]]
        return fitness + distance_matrix[self.path[-1], 0]

    def crossover(self, other):
        length = len(self.path)
        crossover_point = np.random.randint(1, length)
        result = self.path[:crossover_point]
        missing_values = np.setdiff1d(np.arange(1, length), result)
        for val in other.path:
            if val in missing_values:
                result = np.append(result, val)
        return Path(length, result)


r = r0761312()
r.optimize('tour29.csv')
