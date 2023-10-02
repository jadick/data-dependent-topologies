import matplotlib.pyplot as plt
from datasets import load_cifar_10_train
from non_iid_distributions import generate_random_clusters, partition_n_cls, partition_data_dirichlet
from greedy_swap import mean_topology_skew, greedy_swap

def n_cls_experiments(y_train):
    N_cls = [1, 2, 4, 5]
    indx_maps_N_cls = []
    random_10__N_cls = []
    
    for i in N_cls:
        index_map = partition_n_cls(10, i, 100, y_train)
        indx_maps_N_cls.append(index_map)
        random_10__N_cls.append(generate_random_clusters(10, 100, index_map, y_train))
    
    opt_topologies_N_cls = []
    error_logs_N_cls = []
    for random_cluster in random_10__N_cls:
        topology, error_log = greedy_swap(random_cluster, 100)
        opt_topologies_N_cls.append(topology)
        error_logs_N_cls.append(error_log)
    for i in range(4):
        plt.plot(range(0,len(error_logs_N_cls[i])), error_logs_N_cls[i], label = f'n_class = {i+1}')
    plt.ylabel('mean label skew')
    plt.xlabel('step k')
    plt.legend()
    plt.savefig('./results/n_cls.pdf')

def dirichlet_experiments(y_train):
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    maps_Dir = []
    random_10_Dir = []
    
    for alpha in alpha_values:
        index_map = partition_data_dirichlet(100, alpha, y_train)
        maps_Dir.append(index_map)
        random_10_Dir.append(generate_random_clusters(10, 100, index_map, y_train))    

    opt_topologies_Dir = []
    error_logs_Dir = []
    for random_cluster in random_10_Dir:
        topology, error_log = greedy_swap(random_cluster, 100)
        opt_topologies_Dir.append(topology)
        error_logs_Dir.append(error_log)
    for i in range(5):
        plt.plot(range(0,len(error_logs_Dir[i])), error_logs_Dir[0], label= f'alpha = {alpha_values[i]}')
    plt.ylabel('mean label skew')
    plt.xlabel('step k')
    plt.legend()
    plt.savefig('./results/dirichlet.pdf')


cifar10_x_train, cifar10_y_train = load_cifar_10_train()
n_cls_experiments(cifar10_y_train)
dirichlet_experiments(cifar10_y_train)
