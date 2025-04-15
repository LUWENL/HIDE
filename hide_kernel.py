import random

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import time
import math
from itertools import product



def narrow_search_space(N_satellite, N_target, sat_attitudes, sat_occultated):
    sat_sets = []  # shape [N_tar, 可用sat数量]
    for tar_id in range(N_target):
        sat_set = []
        for sat_id in range(N_satellite):
            if (1 - sat_attitudes[sat_id][tar_id]) <= 0.9 and not sat_occultated[sat_id][tar_id]:
                sat_set.append(sat_id)
        sat_sets.append(sat_set)

    return sat_sets


def heuristic_init(sat_sets, tar_prioritys, N_satellite, sat_samplings=None, sat_attitudes=None, sat_motions=None, sort_by='sampling'):

    N_tar = len(tar_prioritys)
    N_sat = N_satellite
    assignment = np.full(N_sat, -1)  # Initialize assignment with -1 (unassigned)

    # Sort targets in descending order of priority
    sorted_targets = np.argsort(-tar_prioritys)

    for target in sorted_targets:
        candidate_sats = sat_sets[target]  # Get candidate satellites for the current target

        if sort_by == 'sampling':
            # Sort candidate satellites by sampling quality in descending order
            sorted_sats = sorted(candidate_sats, key=lambda x: -sat_samplings[x])
        elif sort_by == 'attitude':
            # Sort candidate satellites by attitude factor for the current target in descending order
            sorted_sats = sorted(candidate_sats, key=lambda x: -sat_attitudes[x, target])
        elif sort_by == 'motion':
            # Sort candidate satellites by motion factor for the current target in descending order
            sorted_sats = sorted(candidate_sats, key=lambda x: -sat_motions[x, target])
        else:
            raise ValueError("Invalid sort_by parameter. Choose 'sampling' or 'attitude' or 'motion'.")

        # Assign the highest-ranked available satellite to the target
        for sat in sorted_sats:
            if assignment[sat] == -1:  # Ensure the satellite is not already assigned
                assignment[sat] = target
                break

    solutions = []  # Start with the best solution
    unassigned_sats = np.where(assignment == -1)[0]  # Find unassigned satellites
    if len(unassigned_sats) == 0:
        solutions = [assignment.copy()]
    elif len(unassigned_sats) > 0:
        # Generate all possible combinations of assignments for unassigned satellites
        num_solutions = 10
        for _ in range(num_solutions):  # Generate (num_solutions - 1) additional solutions
            new_assignment = assignment.copy()
            for sat in unassigned_sats:
                # Randomly assign the satellite to a target
                new_assignment[sat] = random.choice(range(N_tar))
            solutions.append(new_assignment)

    return solutions


@cuda.jit
def differential_evolution(new_chromosomes, eliteSize, deRate, de_weight, states1, states2, states3):
    pos = cuda.grid(1)
    F = de_weight[0]

    if eliteSize[0] <= pos < new_chromosomes.shape[0]:
        # mutation
        for a in range(new_chromosomes.shape[1]):

            if xoroshiro128p_uniform_float32(states1, pos) < deRate[0]:
                N_satellite = new_chromosomes.shape[1]
                pos_1 = int(math.floor(xoroshiro128p_uniform_float32(states1, pos) * (N_satellite - 1)))
                pos_2 = int(math.floor(xoroshiro128p_uniform_float32(states2, pos) * (N_satellite - 1)))
                pos_3 = int(math.floor(xoroshiro128p_uniform_float32(states3, pos) * (N_satellite - 1)))

                new_chromosomes[pos][a] = int(
                    new_chromosomes[pos_1][a] + F * (new_chromosomes[pos_2][a] - new_chromosomes[pos_3][a])
                )


@cuda.jit
def destroy_and_repair(chromosomes, chromosomes_for_dar, eliteSize, darRate, states1, states2, states3):
    pos = cuda.grid(1)

    if eliteSize[0] <= pos < chromosomes.shape[0]:
        for a in range(chromosomes.shape[1]):
            if xoroshiro128p_uniform_float32(states1, pos) < darRate[0]:
                N_satellite = chromosomes.shape[1]

                start = int(xoroshiro128p_uniform_float32(states2, pos) * N_satellite)
                length = int(xoroshiro128p_uniform_float32(states3, pos) * (N_satellite // 2)) + 1
                end = min(start + length, N_satellite)

                destroyed_len = end - start
                remaining_len = N_satellite - destroyed_len

                for i in range(start):
                    chromosomes_for_dar[pos][i] = chromosomes[pos][i]
                for i in range(end, N_satellite):
                    chromosomes_for_dar[pos][i - destroyed_len + start] = chromosomes[pos][i]
                for i in range(start, end):
                    chromosomes_for_dar[pos][remaining_len + i - start] = chromosomes[pos][i]

