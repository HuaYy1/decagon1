from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
from itertools import chain
import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

from polypharmacy.utility import (
    load_combo_se,
    load_se_combo,
    load_ppi,
    load_mono_se,
    load_targets,
    load_categories,
)

# Train on CPU (hide GPU) due to memory constraints
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Train on GPU
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders["dropout"]: 0})
    feed_dict.update(
        {placeholders["batch_edge_type_idx"]: minibatch.edge_type2idx[edge_type]}
    )
    feed_dict.update({placeholders["batch_row_edge_type"]: edge_type[0]})
    feed_dict.update({placeholders["batch_col_edge_type"]: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, "Problem 1"

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, "Problem 0"

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        "batch": tf.placeholder(tf.int32, name="batch"),
        "batch_edge_type_idx": tf.placeholder(
            tf.int32, shape=(), name="batch_edge_type_idx"
        ),
        "batch_row_edge_type": tf.placeholder(
            tf.int32, shape=(), name="batch_row_edge_type"
        ),
        "batch_col_edge_type": tf.placeholder(
            tf.int32, shape=(), name="batch_col_edge_type"
        ),
        "degrees": tf.placeholder(tf.int32),
        "dropout": tf.placeholder_with_default(0.0, shape=()),
    }
    placeholders.update(
        {
            "adj_mats_%d,%d,%d" % (i, j, k): tf.sparse_placeholder(tf.float32)
            for i, j in edge_types
            for k in range(edge_types[i, j])
        }
    )
    placeholders.update(
        {"feat_%d" % i: tf.sparse_placeholder(tf.float32) for i, _ in edge_types}
    )
    return placeholders


###########################################################
#
# 加载和预处理数据
#
###########################################################

####
# 以下代码使用人工生成的非常小的网络。
# 由于这些随机网络没有任何有趣的结构，预计性能不会很好。
# main.py的目的是展示如何使用代码！
# 药物组合研究中使用的所有预处理数据集位于：http://snap.stanford.edu/decagon:
# （1）从下载数据集http://snap.stanford.edu/decagon到您的本地机器。
# （2）将此处使用的虚拟玩具数据集替换为您刚刚下载的实际数据集。
# （3）对模型进行训练和测试。
####
combo2stitch, combo2se, se2name = load_combo_se(
    os.path.join("data", "bio-decagon-combo.csv")
)
net, node2idx = load_ppi(os.path.join("data", "bio-decagon-ppi.csv"))
stitch2se, se2name_mono = load_mono_se(os.path.join("data", "bio-decagon-mono.csv"))
stitch2proteins = load_targets(os.path.join("data", "bio-decagon-targets-all.csv"))
se2class, se2name_class = load_categories(
    os.path.join("data", "bio-decagon-effectcategories.csv")
)
se2combo = load_se_combo(os.path.join('data', 'bio-decagon-combo.csv'))

# Number of unique drugs in drug combinations dataset
unique_drugs = set([drug for drug_comb in combo2stitch.values() for drug in drug_comb])

# Creating dictionary from drug to index.
stitch2idx = {node: i for i, node in enumerate(unique_drugs)}
idx2stitch = {i: node for i, node in enumerate(unique_drugs)}

# Creating dictionary from side effects to index.
se2idx = {node: i for i, node in enumerate(list(se2combo.keys()))}
idx2se = {i: node for i, node in enumerate(list(se2combo.keys()))}

val_test_size = 0.05
n_genes = len(node2idx)  # 19081
n_drugs = len(unique_drugs)  # 645
n_drugdrug_rel_types = len(se2combo)  # 1317

# gene_net is a protein-protein interaction networks. We have used PPI data to construct this network.
# The adjacency matrix is a (19081*19081) matrix.
gene_net = net
gene_adj = nx.adjacency_matrix(gene_net)
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()


drug_gene_adj = preprocessing.get_sparse_mat(stitch2proteins, stitch2idx, node2idx)
gene_drug_adj = drug_gene_adj.transpose(copy=True)


drug_drug_adj_list = []

for a, b_assoc in se2combo.items():
    mat = np.zeros((n_drugs, n_drugs))
    for b1, b2 in b_assoc:
        mat[stitch2idx[b1], stitch2idx[b2]] = mat[stitch2idx[b2], stitch2idx[b1]] = 1.
    drug_drug_adj_list.append(sp.csr_matrix(mat))
drug_degrees_list = [
    np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list
]


# data representation
adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_drug_adj],
    (1, 0): [drug_gene_adj],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
}
degrees = {0: [gene_degrees, gene_degrees], 1: drug_degrees_list + drug_degrees_list}

# featureless (genes)
gene_feat = sp.identity(n_genes)
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# features (drugs)
drug_feat = sp.identity(n_drugs)
drug_nonzero_feat, drug_num_feat = drug_feat.shape
drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

# data representation
num_feat = {0: gene_num_feat, 1: drug_num_feat}
nonzero_feat = {0: gene_nonzero_feat, 1: drug_nonzero_feat}
feat = {0: gene_feat, 1: drug_feat}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): "bilinear",
    (0, 1): "bilinear",
    (1, 0): "bilinear",
    (1, 1): "dedicom",
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("neg_sample_size", 1, "Negative sample size.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")
flags.DEFINE_integer("epochs", 50, "Number of epochs to train.")
flags.DEFINE_integer("hidden1", 64, "Number of units in hidden layer 1.")
flags.DEFINE_integer("hidden2", 32, "Number of units in hidden layer 2.")
flags.DEFINE_float("weight_decay", 0, "Weight for L2 loss on embedding matrix.")
flags.DEFINE_float("dropout", 0.1, "Dropout rate (1 - keep probability).")
flags.DEFINE_float("max_margin", 0.1, "Max margin parameter in hinge loss")
flags.DEFINE_integer("batch_size", 512, "minibatch size.")
flags.DEFINE_boolean("bias", True, "Bias term.")
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size,
)

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

print("Create optimizer")
with tf.name_scope("optimizer"):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin,
    )

print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}

###########################################################
#
# Train model
#
###########################################################

print("Train model")
for epoch in range(FLAGS.epochs):

    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict, dropout=FLAGS.dropout, placeholders=placeholders
        )

        t = time.time()

        # Training step: run single weight update
        outs = sess.run(
            [opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict
        )
        train_cost = outs[1]
        batch_edge_type = outs[2]

        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges,
                minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx],
            )

            print(
                "Epoch:",
                "%04d" % (epoch + 1),
                "Iter:",
                "%04d" % (itr + 1),
                "Edge:",
                "%04d" % batch_edge_type,
                "train_loss=",
                "{:.5f}".format(train_cost),
                "val_roc=",
                "{:.5f}".format(val_auc),
                "val_auprc=",
                "{:.5f}".format(val_auprc),
                "val_apk=",
                "{:.5f}".format(val_apk),
                "time=",
                "{:.5f}".format(time.time() - t),
            )

        itr += 1

print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et]
    )
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
    print()
