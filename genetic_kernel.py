from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import time
import math


@cuda.jit
def init_population(chromosomes, N_target, states):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + bx * bw

    if pos < chromosomes.shape[0]:
        for j in range(chromosomes.shape[1]):
            chromosomes[pos][j] = math.floor(xoroshiro128p_uniform_float32(states, pos) * N_target[0])


@cuda.jit()
def eval_genomes_kernel(chromosomes, fitnesses, popSize, N_target, sat_samplings, sat_attitudes, sat_in_darks,
                        sat_vectors, sat_occultated, sat_positions, sat_available, tar_prioritys, tar_vectors, tar_positions):

    # You can re-design this func based on your own requirement.
    pass


@cuda.jit
def sort_chromosomes(chromosomes, fitnesses, sorted_chromosomes, sorted_fitnesses, fitnessTotal):
    pos = cuda.grid(1)
    fitnessTotal[0] = 0

    if pos < chromosomes.shape[0]:
        # print(pos)
        current_fitness = fitnesses[pos][0]
        rank = 0

        for i in range(chromosomes.shape[0]):
            if fitnesses[i][0] > current_fitness:
                rank += 1

        # 处理相同 fitness 的情况
        for i in range(chromosomes.shape[0]):
            if fitnesses[i][0] == current_fitness and i < pos:
                rank += 1

        for j in range(chromosomes.shape[1]):
            sorted_chromosomes[rank][j] = chromosomes[pos][j]
        sorted_fitnesses[rank][0] = current_fitness

        cuda.atomic.add(fitnessTotal, 0, current_fitness)



@cuda.jit
def crossover(new_chromosomes, sorted_chromsomos, sorted_fitnesses, rouletteWheel,
              states, popSize, eliteSize, fitnessTotal, crossoverRate, is_adaptive):
    pos = cuda.grid(1)

    for i in range(0, eliteSize[0]):
        for j in range(sorted_chromsomos.shape[1]):
            # Add new chromosome to next population
            new_chromosomes[i][j] = sorted_chromsomos[i][j]

    for i in range(popSize[0]):
        tmp = sorted_fitnesses[i][0] / fitnessTotal[0]
        if i == 0:
            rouletteWheel[i] = tmp
        else:
            rouletteWheel[i] = rouletteWheel[i - 1] + tmp

    fitness_sum = 0.0
    fitness_max = 0.0

    for i in range(sorted_fitnesses.shape[0]):
        if sorted_fitnesses[i][0] >= fitness_max:
            fitness_max = sorted_fitnesses[i][0]

        fitness_sum += sorted_fitnesses[i][0]

    fitness_avg = fitness_sum / sorted_fitnesses.shape[0]

    # for  i in range(eliteSize, len(chromosomes), 2):
    if eliteSize[0] <= pos < new_chromosomes.shape[0] and pos % 2 == 0:  # 只需要popSize个线程同时处理

        spin1 = xoroshiro128p_uniform_float32(states, pos)
        spin2 = xoroshiro128p_uniform_float32(states, pos)

        j = 0
        for j in range(popSize[0]):
            if rouletteWheel[j] <= spin1:
                j += 1
            else:
                break

        k = 0
        for k in range(popSize[0]):
            if rouletteWheel[k] <= spin2:
                k += 1
            else:
                break

        genome_copy1 = sorted_chromsomos[j]  # Genome of parent 1
        genome_copy2 = sorted_chromsomos[k]  # Genome of parent 2

        if sorted_fitnesses[j][0] >= sorted_fitnesses[k][0]:
            fitness_c = sorted_fitnesses[j][0]
        else:
            fitness_c = sorted_fitnesses[k][0]

        index1 = int(xoroshiro128p_uniform_float32(states, pos) * (len(genome_copy1) - 1))
        index2 = int(xoroshiro128p_uniform_float32(states, pos) * (len(genome_copy1) - 1))

        if is_adaptive[0]:
            crossoverRate_ = crossoverRate[0] * (fitness_max - fitness_c) / (fitness_max - fitness_avg)
            if crossoverRate_ < 0:
                crossoverRate_ = 0.1
            elif crossoverRate_ > 1:
                crossoverRate_ = 1
        else:
            crossoverRate_ = crossoverRate[0]

        if xoroshiro128p_uniform_float32(states, pos) < crossoverRate_:
            for ii in range(len(genome_copy1)):
                if ii <= math.floor(len(genome_copy1) / 2):
                    new_chromosomes[pos][ii] = genome_copy1[(index1 + ii) % len(genome_copy1)]
                    new_chromosomes[pos + 1][ii] = genome_copy2[(index2 + ii) % len(genome_copy2)]
                else:
                    new_chromosomes[pos + 1][ii] = genome_copy1[(index1 + ii) % len(genome_copy1)]
                    new_chromosomes[pos][ii] = genome_copy2[(index2 + ii) % len(genome_copy2)]
        else:
            for ii in range(len(genome_copy1)):
                new_chromosomes[pos][ii] = genome_copy1[ii]
                new_chromosomes[pos + 1][ii] = genome_copy2[ii]


@cuda.jit
def mutation(new_chromosomes, tmp_fitnesses, eliteSize, N_target, states, mutationRate, is_adaptive):
    pos = cuda.grid(1)

    fitness_sum = 0.0
    fitness_max = 0.0
    for i in range(tmp_fitnesses.shape[0]):
        if tmp_fitnesses[i][0] >= fitness_max:
            fitness_max = tmp_fitnesses[i][0]

        fitness_sum += tmp_fitnesses[i][0]

    fitness_avg = fitness_sum / tmp_fitnesses.shape[0]

    if eliteSize[0] <= pos < new_chromosomes.shape[0]:
        fitness_m = tmp_fitnesses[pos][0]
        # mutation
        for a in range(new_chromosomes.shape[1]):
            if is_adaptive[0]:
                mutationRate_ = mutationRate[0] * (fitness_max - fitness_m) / (fitness_max - fitness_avg)
                if mutationRate_ < 0:
                    mutationRate_ = 0.05
                elif mutationRate_ > 1:
                    mutationRate_ = 1
            else:
                mutationRate_ = mutationRate[0]

            if xoroshiro128p_uniform_float32(states, pos) < mutationRate_:
                new_chromosomes[pos][a] = int(
                    math.floor(xoroshiro128p_uniform_float32(states, pos) * (N_target[0] - 1)))
