import numpy as np
import random


def mean_cluster_skew(cluster):
    mean_cluster_skew_vector = 0
    node_label_probabilities = []
    cluster_label_probabilities = []

    for node in cluster:
        label_counts = np.bincount(node[1], minlength = 10)
        label_probabilities = label_counts / len(node[1])
        node_label_probabilities.append(label_probabilities)

    cluster_label_probabilities = np.mean(node_label_probabilities, axis=0)

    for node_probs in node_label_probabilities:
        node_skew = np.absolute(node_probs - cluster_label_probabilities)
        mean_cluster_skew_vector += node_skew

    mean_cluster_skew_vector = mean_cluster_skew_vector / (len(node_label_probabilities[0]))
    mean_cluster_skew = np.mean(mean_cluster_skew_vector)
    return mean_cluster_skew

def mean_topology_skew(topology):
    topology_skew = 0
    for cluster in topology:
        cluster_skew = mean_cluster_skew(cluster)
        topology_skew += cluster_skew
    topology_skew = topology_skew / topology.shape[0]
    return topology_skew

def greedy_swap(clusters, max_steps):
    topology = np.copy(clusters)
    error_log = []
    for k in range(max_steps):
        indx = random.sample(range(0, 9), 2)
        C1 =indx[0]
        C2 = indx[1]

        skew = mean_cluster_skew(topology[C1]) + mean_cluster_skew(topology[C2])
        swaps = []
        swaps_skews = []

        for i in range(len(topology[C1])):
            for j in range(len(topology[C2])):
                temp_i = topology[C1][i]
                topology[C1][i] = topology[C2][j]
                topology[C2][j] = temp_i
                skew_ = mean_cluster_skew(topology[C1]) + mean_cluster_skew(topology[C2])
                topology[C2][j] = topology[C1][i]
                topology[C1][i] = temp_i
                if skew_ < skew:
                    swaps.append((i, j))
                    swaps_skews.append(skew_)

        if len(swaps) > 0:
            print('swaped')
            min_skew = min(swaps_skews)
            chosen_swap = swaps[np.where(swaps_skews == min_skew)[0][0]]
            temp_i = topology[C1][chosen_swap[0]]
            topology[C1][chosen_swap[0]] = topology[C2][chosen_swap[1]]
            topology[C2][chosen_swap[1]] = temp_i

        topology_skew = mean_topology_skew(topology)
        error_log.append(topology_skew)
        print(k, topology_skew)

    return topology, error_log
