from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN


def get_model(config, adj):
    if config['model'] == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif config['model'] == 'gcn_cheby':
        support = chebyshev_polynomials(adj, config['max_degree'])
        num_supports = 1 + config['max_degree']
        model_func = GCN
    elif config['model'] == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(config['model']))
    return support, num_supports, model_func


# Define model evaluation function
def evaluate(model, sess, features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


config = {}
config['hidden1'] = 16
config['learning_rate'] = 0.01
config['weight_decay'] = 5e-4
config['max_degree'] = 3
config['early_stopping'] = 10
config['dropout'] = 0.5
config['epochs'] = 200
config['model'] = 'gcn'
config['dataset'] = 'cora'


def main(model_config, sess, seed, verbose=False):
    # print config
    very_beginning = time.time()
    print(model_config)

    # load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_test = load_data(model_config['dataset'])

    data_split = {
        'adj': adj,
        'features': features,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'idx_test': idx_test
    }

    # Some preprocessing  # features = (coords, values, shape)
    begin = time.time()
    print(time.time() - begin, 's')
    features = preprocess_features(features)

    #
    support, num_supports, model_func = get_model(model_config, adj)

    # define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # create model
    model = model_func(model_config, placeholders, input_dim=features[2][1], logging=True)

    # random initialize
    sess.run(tf.global_variables_initializer())

    # train
    cost_val = []
    for epoch in range(config['epochs']):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: model_config['dropout']})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(model, sess, features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        if verbose:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > model_config['early_stopping'] and cost_val[-1] > np.mean(cost_val[-(model_config['early_stopping'] + 1):-1]):
            print("Early stopping...")
            break

    print("---Optimization Finished!---")

    # Testing
    # test_cost, test_acc, test_duration = evaluate(model, sess, features, support, y_test, test_mask, placeholders)
    # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    #
    t_test = time.time()
    feed_dict_test = construct_feed_dict(features, support, y_test, test_mask, placeholders)
    test_cost, test_acc, test_acc_all, test_o_acc_all = sess.run([model.loss, model.accuracy, model.accuracy_all, model.o_accuracy_all], feed_dict=feed_dict_test)
    test_duration = (time.time() - t_test)
    print('test_mask: ', test_mask)
    print('test_acc_all', test_acc_all)
    print('test_o_acc_all', test_o_acc_all)

    print("Total time={}s".format(time.time()-very_beginning))
    return test_acc, test_acc_all, test_o_acc_all, data_split


if __name__ == '__main__':

    seed = 123
    np.random.seed(seed)

    # init session
    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        with tf.Session() as sess:
            test_acc, test_acc_all, test_o_acc_all, data_split = main(config, sess, seed)
