# This file contains helper methods for the membraneProtML project in order to keep the main python files readable
# Author: Eric Boone
# Date: 10/3/2022

import numpy as np
import pandas as pd
import torch

# TODO: This will have to change when I fix the implemtation of SM2 in the kernel

AA_letter_to_index = {"A":1,
                      "R":2,
                      "N":3,
                      "D":4,
                      "C":5,
                      "Q":6,
                      "E":7,
                      "G":8,
                      "H":9,
                      "I":10,
                      "L":11,
                      "K":12,
                      "M":13,
                      "F":14,
                      "P":15,
                      "S":16,
                      "T":17,
                      "W":18,
                      "Y":19,
                      "V":20,
                      "B":21,
                      "J":22,
                      "Z":23,
                      "X":24,
                      "*":25,
                      "U":26
                      }


def read_data(filename):
    data_matrix = pd.read_csv(filename, sep=',', header=None)
    return data_matrix.iloc[:, 2], data_matrix.iloc[:, 0]


def int_labels_to_onehot(vector: np.array):
    """
    Turns labels from integers (i.e. 0, 1, 2) to one-hot vectors (i.e. [1,0,0], [0,1,0],[0,0,1]). Assumes integer labels
    start at 0.
    :param vector:
    :return: matrix of one-hot vectors representing labels.
    """
    num_labels = np.max(vector) + 1
    onehots = np.zeros((len(vector), num_labels))

    for i in range(len(vector)):
        onehots[i][vector[i]] = 1

    return onehots


def sequence_to_tensor(seq, ntokens):
    seq_len = len(seq)
    onehot_mat = np.zeros((seq_len, ntokens), int)

    for i in range(seq_len):
        onehot_mat[i][AA_letter_to_index[seq[i]]-1] = 1

    return torch.tensor(onehot_mat)


def calc_selfdisctance_approx(k1, S):
    k1_identical_av = np.average(np.diag(k1))
    n = len(S)

    k_mer_scores = np.power(np.repeat(k1_identical_av, n), range(n,0,-1))
    coefficients = np.arange(1,n+1)

    return np.einsum('i,i->', k_mer_scores, coefficients)


# Used to double check requirements of SM2 in kernel function
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

