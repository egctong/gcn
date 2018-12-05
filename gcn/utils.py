import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("gcn/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        # only fill with data those indexes which are part of the test idx range!
        # if they aren't, they're just left with zeros.
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # node groups:
    node_per_cpn = get_node_cpn(graph)
    nodes_per_group = get_node_groups(graph)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test, nodes_per_group, node_per_cpn


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def get_node_groups(graph):
    gr = nx.from_dict_of_lists(graph)
    acs = sorted(nx.connected_components(gr), key=len, reverse=True)
    nodes_in_biggest = acs[0]

    nodes_per_group = {}
    for c in acs[1:]:
        if len(c) in nodes_per_group:
            nodes_per_group[len(c)] = nodes_per_group[len(c)].union(c)
        else:
            nodes_per_group[len(c)] = c
    nodes_per_group[0] = nodes_in_biggest
    return nodes_per_group


def get_node_cpn(graph):
    gr = nx.from_dict_of_lists(graph)

    acs = sorted(nx.connected_components(gr), key=len, reverse=True)

    nodes_per_cpn = {}
    tmp = {len(c): 0 for c in acs}

    for c in acs[1:]:
        c_size = len(c)
        tmp[c_size] += 1
        c_label = tmp[c_size] - 1
        c_name = '{}_{}'.format(c_size, c_label)

        nodes_per_cpn[c_name] = c

    nodes_per_cpn['0_0'] = acs[0]

    return nodes_per_cpn


def performance_per_group(nodes_per_group, nodes_per_cpn, test_o_acc_all, data_split, the_rest=False):
    """
    :param nodes_per_group: in the entire graph, component size: all nodes in components of that size (for largest component, size =0)
    :param test_o_acc_all: original test accuracy for all nodes in the graph (before masked)
    :param data_split:
    :return:
    """

    #list of only the test nodes in the graph
    idx_test = data_split['idx_test']

    # accuracy for each of the tested nodes
    test_per_node = {}
    for node in idx_test:
        test_per_node[node] = test_o_acc_all[node]

    # for each component size, what is the mean accuracy of the tested nodes

    cpn_sizes = list(set([cname[0] for cname in nodes_per_cpn]))

    print('========= BIGGEST CPN OR NOT===========')
    acc_of_B = np.mean([test_per_node[node] for node in idx_test if node in nodes_per_group[0]])
    print('test accuracy for BIGGEST = ', acc_of_B)
    acc_of_nonB = np.mean([test_per_node[node] for node in idx_test if node not in nodes_per_group[0]])
    print('test accuracy for the rest = ', acc_of_nonB)

    if the_rest:
        print('=========per SIZE===========')

        for phase in ['train', 'val', 'test']:
            idx = data_split['idx_' + phase]
            print('----{} (TOT = )----'.format(phase, len(idx)))
            for cpn_size in nodes_per_group:
                num_in_phase = len([n for n in nodes_per_group[cpn_size] if n in idx])
                print('there are {} nodes coming from components of size {} ({}% of train set)'.format(num_in_phase, cpn_size, num_in_phase/ len(idx)*100))
                if phase == 'test':
                    acc_of_cpn_size = np.mean([test_per_node[node] for node in idx_test if node in nodes_per_group[cpn_size]])
                    print('test accuracy for these nodes = ', acc_of_cpn_size)

        print('==============CPN ===========')
        for cpn in nodes_per_cpn:
            print('-----{}-----'.format(cpn))
            for phase in ['train', 'val', 'test']:
                idx = data_split['idx_' + phase]
                num_in_phase = len([n for n in nodes_per_cpn[cpn] if n in idx])
                print('{} : {} ({}%)'.format(phase, num_in_phase, num_in_phase/ len(nodes_per_cpn[cpn])))

            acc_of_cpn = np.mean([test_per_node[node] for node in idx_test if node in nodes_per_cpn[cpn]])
            print('test accuracy = ', acc_of_cpn)

    # acc_per_group = {}
    # for cpn_size in nodes_per_group:
    #     acc_per_group[cpn_size] = [test_per_node[node] for node in idx_test if node in nodes_per_group[cpn_size]]
    #     acc_per_group[cpn_size] = np.mean(acc_per_group[cpn_size])
    #     print('components of size = {}, number of nodes = {} ({}%), accuracy = {}'.format(cpn_size,
    #                                                                                     len(test_per_node[cpn_size]),
    #                                                                                     len(test_per_node[cpn_size])/len(idx_test)*100,
    #                                                                                     acc_per_group[cpn_size]))
    #
    return test_per_node, acc_of_B, acc_of_nonB