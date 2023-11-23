import random
import numpy as np

from src.models.member import Member
from src.models.population import Population


def parent_centric_mutation(population: Population):
    new_members = []
    for _ in range(population.size):
        selected_members = np.array(random.sample(population.members.tolist(), 3))
        new_member = parent_centric_crossover(selected_members)

        new_members.append(new_member)

    new_population = Population(
        interval=population.interval,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def perpendicular_distance(line_point1, line_point2, point):
    line_point1 = np.array(line_point1)
    line_point2 = np.array(line_point2)
    point = np.array(point)

    line_vector = line_point2 - line_point1
    point_vector = point - line_point1

    projection = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    projected_point = line_point1 + projection * line_vector

    distance = np.linalg.norm(point - projected_point)
    return distance


def parent_centric_crossover(parents, sigma_s=1.0, sigma_eta=1.0):
    parents_arr = []
    for p in parents:
        parents_arr.append(np.array([ch.real_value for ch in p.chromosomes]))
    parents_arr = np.array(parents_arr)

    g = np.mean(parents_arr, axis=0)  # Compute the mean vector
    p_index = np.random.randint(len(parents))  # Select a parent at random
    xp = parents_arr[p_index]  # Selected parent

    dp = xp - g  # Direction vector dp→ = xp,G - g⃗

    remaining_parents = np.delete(parents_arr, p_index, axis=0)  # Remove the selected parent from the list

    perpendicular_distances = np.array([perpendicular_distance(g, xp, remaining_parents[i]) for i in range(len(remaining_parents))])
    # perpendicular_distances = np.abs(np.dot(remaining_parents - g, dp)) / np.linalg.norm(dp)
    D_bar = np.mean(perpendicular_distances)

    e = np.linalg.qr(dp.reshape(1, -1).T, mode='complete')[0][:, 1:]  # Orthonormal bases e⃗ i

    w_s = np.random.normal(0, sigma_s)  # Zero mean normally distributed variable wς with variance σ2ς
    w_eta = np.random.normal(0, sigma_eta, size=e.shape[1])  # Zero mean normally distributed variable wη with variance σ2η

    offspring = xp + w_s * dp + np.dot(w_eta * D_bar, e.T)  # Generate offspring

    new_member = Member(parents[0].interval, parents[0].args_num)
    for i in range(parents[0].args_num):
        new_member.chromosomes[i].real_value = offspring[i]
    return new_member
