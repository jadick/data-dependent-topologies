import numpy as np
import random
from datasets import  load_cifar_10_train,load_cifar_10_test

def generate_random_clusters(n_clusters, n_nodes, index_map, y_train):

    clusters = np.zeros([n_clusters, n_nodes // n_clusters, 2 , len(y_train) // n_nodes]).astype(int)
    #cluster_array[cluster][node][id/label][local dataset index]
    nodes_ids = list(range(0, n_nodes))

    for cluster in range(clusters.shape[0]):
        cluster_nodes_list = []
        for node in range(clusters.shape[1]):
            np.random.shuffle(nodes_ids)
            node_id = nodes_ids.pop()
            data_pairs = np.transpose(np.asarray([[data_indx, y_train[data_indx]] for data_indx in index_map[node_id]]))
            clusters[cluster][node] = data_pairs
    return clusters

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        net_cls_counts.append (tmp)
    return net_cls_counts


def partition_n_cls(n_cls_dataset, n_cls_node, n_nodes, y_train):
    size_shard = (len(y_train) // n_nodes) // n_cls_node
    n_shards_per_class = (len(y_train) // n_cls_dataset) // size_shard
    sorted_classes = [] #np.zeros((10,5000), dtype=int)
    shards = []

    net_dataidx_map = {j:[] for j in range(n_nodes)}

    for i in range(10): #sort data per label
        sorted_classes.append(np.where(y_train == i)[0].tolist())

    for cls in range(n_cls_dataset): # generate 'n_shards_per_class' for each 'n_cls_dataset'
        cls_shards = []
        for cls_shard in range(n_shards_per_class):
            shard = []
            for shard_data in range(size_shard):
                rand_indx = random.randint(0,len(sorted_classes[cls])-1)
                shard.append(sorted_classes[cls].pop(rand_indx))
            cls_shards.append(shard)
        shards.append(cls_shards)
    shards_flat = [shard for cls in shards for shard in cls]

    for node in net_dataidx_map.keys():
        for cls in range(n_cls_node):
            rand_indx = random.randint(0,len(shards_flat)-1)
            net_dataidx_map[node] += shards_flat.pop(rand_indx)

    return net_dataidx_map

def correct_missing_idx(y_train, idx_map, cls_counts):
    indx_map_copy = idx_map.copy()
    cls_counts_copy = cls_counts.copy()
    missing_data_map = np.ones(y_train.shape)
    for key, value in indx_map_copy.items():
        for index in value:
            missing_data_map[index] = 0
    # missing values = 1, not missing =0

    data_per_node = np.sum(cls_counts_copy,axis=1)
    missing_data_per_node = [500 - node for node in data_per_node]

    missing_indx = []
    for idx in range(0, 50000):
        if missing_data_map[idx] == 1:
            missing_indx.append(idx)


    for node in range(len(missing_data_per_node)):
        while missing_data_per_node[node] > 0:
            np.random.shuffle(missing_indx)
            index = missing_indx.pop()
            indx_map_copy[node].append(index)
            missing_data_per_node[node] -= 1
            cls_counts_copy[node][y_train[index]] += 1

    return cls_counts_copy, indx_map_copy

def partition_data_dirichlet(n_nodes, alpha, y_train):
    n_train = y_train.shape[0]
    n_cls = 10

    n_data_per_clnt = len(y_train) / n_nodes
    clnt_data_list = ([n_train//n_nodes] * n_nodes)
    cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_nodes)
    idx_list = [np.where(y_train == i)[0].tolist() for i in range(n_cls)]
    cls_amount = [len(idx_list[i]) for i in range(n_cls)]
    net_dataidx_map= {j: [] for j in range(n_nodes)}

    while np.sum(clnt_data_list) != 1000:
        while True:
            rand_node = np.random.randint(n_nodes) # select a random node
            if clnt_data_list[rand_node] <= 0:
                continue
            clnt_data_list[rand_node] -= 1
            dir_prior = cls_priors[rand_node]
            multinomial_sample = np.random.multinomial(1, dir_prior)
            class_sample = np.where(multinomial_sample == 1)[0][0]

            if cls_amount[class_sample] <= 0:
                clnt_data_list[rand_node] += 1
                continue
            cls_amount[class_sample] -= 1
            rand_class_idx = np.random.randint(len(idx_list[class_sample]))
            selected_data_idx = idx_list[class_sample].pop(rand_class_idx)
            net_dataidx_map[rand_node].append(selected_data_idx)

            #print(np.sum(cls_amount), np.sum(clnt_data_list))
            break

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    traindata_cls_counts, net_dataidx_map = correct_missing_idx(y_train, net_dataidx_map, traindata_cls_counts)
    return net_dataidx_map
