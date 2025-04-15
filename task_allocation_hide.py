import time
import random
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from genetic_kernel import *
from run_SAC import METADATA
from hide_kernel import heuristic_init, narrow_search_space, differential_evolution, destroy_and_repair


def hide_kernel(N_satellite, N_target, sat_samplings, sat_attitudes, sat_motions, sat_in_darks,
                sat_vectors, sat_occultated, sat_positions, sat_available, tar_prioritys, tar_vectors, tar_positions):
    popSize = METADATA['popSize']
    eliteSize = METADATA['eliteSize']
    chrom_size = N_satellite
    # crossoverRate = METADATA['crossoverRate']
    deRate = METADATA['deRate']
    darRate = METADATA['darRate']
    crossoverRate = 1 - METADATA['deRate']
    de_weight = METADATA['de_weight']
    mutationRate = METADATA['mutationRate']
    num_generations = METADATA['generations']

    #  input array
    cuda_chromosomes = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_chromosomes_for_dar = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_sorted_chromosomes = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_sorted_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_tmp_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))  # only for mutation
    cuda_fitnessTotal = cuda.to_device(np.zeros(shape=1, dtype=np.float64))
    cuda_rouletteWheel = cuda.to_device(np.zeros(shape=popSize, dtype=np.float64))
    cuda_popSize = cuda.to_device(np.array([popSize], dtype=np.int32))
    cuda_eliteSize = cuda.to_device(np.array([eliteSize], dtype=np.int32))
    cuda_crossoverRate = cuda.to_device(np.array([crossoverRate], dtype=np.float64))
    cuda_mutationRate = cuda.to_device(np.array([mutationRate], dtype=np.float64))
    cuda_deRate = cuda.to_device(np.array([deRate], dtype=np.float64))
    cuda_darRate = cuda.to_device(np.array([darRate], dtype=np.float64))
    cuda_de_weight = cuda.to_device(np.array([de_weight], dtype=np.float64))
    cuda_N_target = cuda.to_device(np.array([N_target], dtype=np.int32))
    cuda_sat_samplings = cuda.to_device(sat_samplings)
    cuda_sat_attitudes = cuda.to_device(sat_attitudes)
    cuda_sat_in_darks = cuda.to_device(sat_in_darks)
    cuda_sat_vectors = cuda.to_device(sat_vectors)
    cuda_sat_positions = cuda.to_device(sat_positions)
    cuda_sat_available = cuda.to_device(sat_available)
    cuda_sat_occultated = cuda.to_device(sat_occultated)
    cuda_tar_prioritys = cuda.to_device(tar_prioritys)
    cuda_tar_vectors = cuda.to_device(tar_vectors)
    cuda_tar_positions = cuda.to_device(tar_positions)
    cuda_is_adaptive = cuda.to_device(np.array([METADATA['adaptive']]))

    start = time.perf_counter()

    threads_per_block = 32
    blocks_per_grid = (popSize + (threads_per_block - 1)) // threads_per_block
    # print(threads_per_block, blocks_per_grid)
    blocks_per_grid, threads_per_block = 64, 32

    # states
    np.random.seed(METADATA['seed'])
    random.seed(METADATA['seed'])

    state_seeds = np.random.rand(9)
    states = []
    for i in range(len(state_seeds)):
        states.append(create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=state_seeds[i]))

    if METADATA['init'] == 'hybrid_init':
        init_population[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_N_target, states[0])

        # narrow the search space
        sat_sets = narrow_search_space(N_satellite, N_target, sat_attitudes, sat_occultated)

        # heuristic init
        heuristic_solution1 = heuristic_init(sat_sets, tar_prioritys, N_satellite, sat_samplings=sat_samplings, sat_attitudes=None, sat_motions=None, sort_by='sampling')
        heuristic_solution2 = heuristic_init(sat_sets, tar_prioritys, N_satellite, sat_samplings=None, sat_attitudes=sat_attitudes, sat_motions=None, sort_by='attitude')
        heuristic_solution3 = heuristic_init(sat_sets, tar_prioritys, N_satellite, sat_samplings=None, sat_attitudes=None, sat_motions=sat_motions, sort_by='motion')

        all_len = len(heuristic_solution1) + len(heuristic_solution2) + len(heuristic_solution3)

        cuda_hi_chromosomes = cuda.to_device(np.concatenate([heuristic_solution1, heuristic_solution2, heuristic_solution3]))

        cuda_chromosomes[:all_len] = cuda_hi_chromosomes[:all_len]

        # si: sorted init
        si_popSize = METADATA['popSize'] * 10000
        # si_popSize = int(1e6)
        si_cut_popSize = int(METADATA['popSize'] * 0.8) - all_len
        cuda_si_chromosomes = cuda.to_device(np.zeros([si_popSize, chrom_size], dtype=np.int32))
        cuda_si_fitnesses = cuda.to_device(np.zeros([si_popSize, 1], dtype=np.float64))
        cuda_si_popSize = cuda.to_device(np.array([si_popSize], dtype=np.int32))
        cuda_sorted_si_chromosomes = cuda.to_device(np.zeros([si_popSize, chrom_size], dtype=np.int32))
        cuda_sorted_si_fitnesses = cuda.to_device(np.zeros([si_popSize, 1], dtype=np.float64))
        cuda_si_fitnessTotal = cuda.to_device(np.zeros(shape=1, dtype=np.float64))

        init_population[blocks_per_grid, threads_per_block](cuda_si_chromosomes, cuda_N_target, states[0])
        eval_genomes_kernel[blocks_per_grid, threads_per_block](cuda_si_chromosomes, cuda_si_fitnesses, cuda_si_popSize, cuda_N_target, cuda_sat_samplings, cuda_sat_attitudes,
                                                                cuda_sat_in_darks, cuda_sat_vectors, cuda_sat_occultated, cuda_sat_positions, cuda_sat_available,
                                                                cuda_tar_prioritys, cuda_tar_vectors, cuda_tar_positions)
        sort_chromosomes[blocks_per_grid, threads_per_block](cuda_si_chromosomes, cuda_si_fitnesses, cuda_sorted_si_chromosomes,
                                                             cuda_sorted_si_fitnesses, cuda_si_fitnessTotal)

        cuda_chromosomes[all_len: all_len + si_cut_popSize] = cuda_sorted_si_chromosomes[:si_cut_popSize]
        cuda_fitnesses[all_len: all_len + si_cut_popSize] = cuda_sorted_si_fitnesses[:si_cut_popSize]

    else:
        init_population[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_N_target, states[0])

    for i in range(num_generations + 1):

        eval_genomes_kernel[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses, cuda_popSize, cuda_N_target, cuda_sat_samplings, cuda_sat_attitudes,
                                                                cuda_sat_in_darks, cuda_sat_vectors, cuda_sat_occultated, cuda_sat_positions, cuda_sat_available,
                                                                cuda_tar_prioritys, cuda_tar_vectors, cuda_tar_positions)


        # if i == 0:
        #     fitnesses = cuda_fitnesses.copy_to_host()
        # print("max fitness of heuristic init: ", np.max(fitnesses[:all_len]) )
        # print("max fitness of sorted init: ", np.max(fitnesses[all_len: all_len + si_cut_popSize]))
        # print("max fitness of hybrid init: ", np.max(cuda_fitnesses.copy_to_host()))

        # for Ablation Experiment of Hybrid Initialization
        # record = [np.max(fitnesses[:all_len]), np.max(fitnesses[all_len: all_len + si_cut_popSize])]
        # record = [np.max(fitnesses)]

        if i < num_generations:

            sort_chromosomes[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses,
                                                                 cuda_sorted_chromosomes,
                                                                 cuda_sorted_fitnesses, cuda_fitnessTotal)

            # print("{} th / 100".format(i), np.max(cuda_fitnesses.copy_to_host()), cuda_fitnessTotal.copy_to_host())
            # print(cuda_chromosomes.copy_to_host()[:5])
            # print(cuda_fitnesses.copy_to_host()[:5])

            # Crossover And Mutation
            crossover[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_sorted_chromosomes,
                                                          cuda_sorted_fitnesses, cuda_rouletteWheel,
                                                          states[1], cuda_popSize, cuda_eliteSize, cuda_fitnessTotal,
                                                          cuda_crossoverRate, cuda_is_adaptive)

            differential_evolution[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_eliteSize, cuda_deRate, cuda_de_weight,
                                                                       states[3], states[4], states[5])


            mutation[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_tmp_fitnesses, cuda_eliteSize, cuda_N_target,
                                                         states[2], cuda_mutationRate, cuda_is_adaptive)

            # # destroy and repair (dar)
            cuda_chromosomes_for_dar = cuda_chromosomes
            destroy_and_repair[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_chromosomes_for_dar, cuda_eliteSize, cuda_darRate, states[6], states[7], states[8])
            cuda_chromosomes = cuda_chromosomes_for_dar


    end = time.perf_counter()

    show_fitness_pairs = []
    chromosomes = cuda_chromosomes.copy_to_host()
    fitnesses = cuda_fitnesses.copy_to_host()
    # print(fitnesses)

    for i in range(len(chromosomes)):
        show_fitness_pairs.append([chromosomes[i], fitnesses[i]])
    fitnesses = list(reversed(sorted(fitnesses)))  # fitnesses now in descending order
    show_sorted_pairs = list(reversed(sorted(show_fitness_pairs, key=lambda x: x[1])))
    best_allocation = show_sorted_pairs[0][0]
    best_fitness = show_sorted_pairs[0][1]

    # record.append(best_fitness[0])
    # print(record, ",")

    return best_allocation, best_fitness[0], end - start
