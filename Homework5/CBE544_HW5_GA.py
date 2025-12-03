#!/home/users/seunghh/anaconda3/envs/pormake/bin/python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
import sys
import copy
import random
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prod(iterable):
    p = 1
    for v in iterable:
        p *= v
    return p


def pick(list_like):
    return random.sample(list_like, 1)[0]


class GeneticAlgorithm:
    def __init__(self, topology, possible_node_bbs, possible_edge_bbs):
        self.topology = topology
        self.possible_node_bbs = possible_node_bbs
        self.possible_edge_bbs = possible_edge_bbs

        self.n_possible_combinations = \
            prod([len(v) for v in possible_node_bbs]) \
            * len(possible_edge_bbs)**topology.n_edge_types

        self.chromo_length = topology.n_node_types + topology.n_edge_types

    def make_random_chromo(self):
        chromo = []
        for i in range(self.topology.n_node_types):
            bb = pick(self.possible_node_bbs[i])
            chromo.append(bb)

        for i in range(self.topology.n_edge_types):
            bb = pick(self.possible_edge_bbs)
            chromo.append(bb)

        return chromo

    def chromo2key(self, chromo):
        key = self.topology.name + "+" + "+".join(chromo)
        return key

    def tournament_selection(self, population, fs, p=0.9):
        """
        fs: fitness values.
        """
        n = len(population)
        i, j = np.random.choice(n, size=2, replace=False)

        # Enforce fs[i] >= fs[j].
        if fs[i] > fs[j]:
            i, j = j, i

        if np.random.rand() < p:
            chromo = population[i]
        else:
            chromo = population[j]

        return chromo

    def uniform_crossover(self, chromo_i, chromo_j):
        p = np.random.rand(self.chromo_length)
        child = np.where(p < 0.5, chromo_i, chromo_j).tolist()

        return child

    def mutation(self, chromo):
        chromo = copy.deepcopy(chromo)
        i = np.random.randint(low=0, high=self.chromo_length)
        if i < self.topology.n_node_types:
            # Mutate nodes.
            chromo[i] = pick(self.possible_node_bbs[i])
        else:
            # Mutate edges.
            chromo[i] = pick(self.possible_edge_bbs)

        return chromo

    def brute_force_search(self, fitness_function):
        """
        Run this method when
        population_size*n_generation > n_possible_combinations.
        """
        iters = []

        for i in range(self.topology.n_node_types):
            iters.append(iter(self.possible_node_bbs[i]))

        for i in range(self.topology.n_edge_types):
            iters.append(iter(self.possible_edge_bbs))

        saved_fitness_values = {}
        keys = []
        for bbs in itertools.product(*iters):
            if len(keys) == 1000:
                fs = fitness_function(keys)
                #print(fs)
                for k, v in zip(keys, fs):
                    saved_fitness_values[k] = v
                keys = []
                print(k, v)

            key = self.topology.name + "+" + "+".join(bbs)
            keys.append(key)

        # Calculate fitness of rest keys.
        if keys:
            fs = fitness_function(keys)
            for k, v in zip(keys, fs):
                saved_fitness_values[k] = v

        return saved_fitness_values

    def run(self,
            fitness_function,
            mutation_prob=0.2,
            population_size=1000,
            n_generation=100,
            fitness_data=None,
            plot_hist=False,
            print_state=False):
        if population_size*n_generation >= self.n_possible_combinations:
            print("brute force search starts.")
            return self.brute_force_search(fitness_function)

        population = [
            self.make_random_chromo() for _ in range(population_size)]

        fitness_data_count = 0
        saved_fitness_values = {}
        for i in range(n_generation):
            keys = [self.chromo2key(chromo) for chromo in population]
            fs = fitness_function(keys)

            # Use pre-calculated fitness values if available.
            if fitness_data is not None:
                for j, key in enumerate(keys):
                    if key in fitness_data.index:
                        fs[j] = fitness_data.loc[key]
                        fitness_data_count += 1

            for k, v in zip(keys, fs):
                saved_fitness_values[k] = v

            # Take top 10 for next generation.
            sorted_population_with_fs = \
                sorted([v for v in zip(fs, population)])
            sorted_population = [v for _, v in sorted_population_with_fs]

            new_population = sorted_population[:10]
            duplication_check_set = \
                set([self.chromo2key(c) for c in new_population])

            if plot_hist:
                plt.hist(fs, bins=201)
                plt.show()

            if print_state:
                print("Generation Index:", i,
                      "Generated MOFs:", len(saved_fitness_values))
                if fitness_data is not None:
                    print("# of fitness data eval:", fitness_data_count)
                print(sorted_population_with_fs[:5])

            #new_population = []
            #duplication_check_set = set()
            while len(new_population) < population_size:
                chromo_i = self.tournament_selection(population, fs, p=0.9)
                chromo_j = self.tournament_selection(population, fs, p=0.9)

                if np.random.rand() < 0.9:
                    chromo = self.uniform_crossover(chromo_i, chromo_j)
                else:
                    chromo = pick([chromo_i, chromo_j])

                if np.random.rand() < mutation_prob:
                    chromo = self.mutation(chromo)

                key = self.chromo2key(chromo)
                if key in duplication_check_set:
                    continue

                new_population.append(chromo)
                duplication_check_set.add(key)

            population = new_population

        return saved_fitness_values
