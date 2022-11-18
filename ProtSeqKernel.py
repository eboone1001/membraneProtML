
# Protein Sequence Kernel: The following file contains the functions for the calculation of the kernal distance between
# two protein sequences for use in a non-linear SVM.  The kernal function used here is an implementation of the kernel
# proposed in:

# Nojoomi, S., Koehl, P. String kernels for protein sequence comparisons: improved fold recognition.
# BMC Bioinformatics 18, 137 (2017). https://doi.org/10.1186/s12859-017-1560-9

# Author: Eric Boone
import math

import blosum
import numpy as np
import membraneProtML_helper as hlpr

k1_index = hlpr.AA_letter_to_index

def get_sm():
    # TODO: Skipped this step for ease. SM is a "raw counts blosum matrix". For now I am using BLOSUM62
    return False


def get_sm2():
    # TODO: Existing implementation is VERY dirty. This is the first thing to fix.
    """ First step is to calculate the SM2 matrix as defined in the paper
    SM2 is pointed out to be round(log2(blosum)) so here I've just grabbed
    BLOSUM62 and found the exp2() of it. Unfortunately I cant undo the rounding. """

    blosum62_values = list(blosum.BLOSUM(62).values())
    blosum62_matrix = np.reshape(blosum62_values, (25, 25))
    sm2 = np.exp2(blosum62_matrix)

    # Double check:
    try:
        np.linalg.cholesky(sm2)
        assert hlpr.check_symmetric(sm2)
    except np.linalg.LinAlgError:
        raise IOError("SM2 is not positive definite")
    except AssertionError:
        raise IOError("SM2 is not symmetric")

    return sm2


def calc_k1_matrix(sm2, beta=.2):
    # TODO: add k1 function that allows for the user to get K1 scores by inputing CHARS and avoid the mess in calc_k2
    """
    The first kernel describes the distance between any two amino acids. Really just an exponentiated BLOSUM, but will
    used to calculate later kernels. For ease of programming, I have choosen to output K1 as a matrix that can then be
    indexed and passed to later kernels, instead of making it it's own function.
    :param sm2:
    :return: matrix the same size as SM2
    """
    k1 = np.power(sm2, beta)
    return k1


def calc_k2(k1, u, v):
    """
    This kernel is a step-up from K1, and allows us to calculate the distance between two k-mers.
    :param k1: first kernel; output of calc_k1()
    :param u, v: two strings of length k
    :return: K2(u,v) or an integer representing the distance between the input
    """

    if len(u) != len(v):
        raise IOError("k-mers are not of the same length")

    product = 1

    for i in range(0,len(u)):

        ind1 = k1_index[u[i]]
        ind2 = k1_index[v[i]]
        # print(u[i], ind1, " : ",u[i], ind2)
        product = product * k1[ind1, ind2]

    return product


def calc_k3(k1, S, T, k_max=10):
    """
    This is a dynamically programed implementation of the K3 kernel that gives a calculated distance between two protein
     sequences. Results in a speed-up of about 10 times on real protein sequences.
    :param k1: first kernel; output of calc_k1()
    :param S, T: two protein sequences of any length
    :return: k3: a number representing the similarity score of two seq.
    """

    # let len(S) and len(t), be known as m and n, respectively for convenience
    m = len(S)
    n = len(T)

    # Set up the dynamic programming array.
    k_layer = np.zeros((m, n))
    total_score = 0

    # Set base case
    for i in range(0, m):
        for j in range(0, n):
            k_layer[i][j] = calc_k2(k1, S[i], T[j])

    base_layer = k_layer

    for k in range(1, min(k_max, m, n)):
        k_minus_one_layer = k_layer
        k_layer = np.zeros((m - k, n - k))
        for i in range(0, m-k):
            for j in range(0, n-k):
                k_layer[i][j] = k_minus_one_layer[i][j] * base_layer[i+k][j+k]
                #print(k_minus_one_layer)
                #print(k_layer)
        total_score += np.sum(k_layer)
        # base_layer = base_layer[1:][1:]

    return total_score


def calc_k3_correlation(k1, S, T, k_max=10):
    """
    This is a function that takes the output of K3, and accounts for the fact that sums automatically give higher scores
    to longer sequences. Also means that 0 < K3(S,T) < 1.
    :param k1: first kernel; output of calc_k1()
    :param S, T: two protein sequences of any length
    :return: k3
    """
    # TODO: This is still not fast enough. The killer is on K3(S,S) when S is a long AA seq. So far my best solution is
    # TODO: to divide R3(S,T) by the summed length of S and T. This should be OK for now as long as k3_corr does not
    # TODO: need to be less than 1, and even then, I am going to have to use a much smaller dataset.

    # This is the proper implementation, for now, it takes FAR to long to be used, especially with the current dataset
    S_k3 = calc_k3(k1, S, S)
    T_k3 = calc_k3(k1, T, T)
    ST_k3 = calc_k3(k1, S, T)

    #k3_corr = ST_k3/(math.sqrt(S_k3*T_k3))

    # For now, I'll use this much simpler version, using length instead of score..
    """ ST_k3 = calc_k3(k1, S, T)
    S_k3_approx = hlpr.calc_selfdisctance_approx(k1, S)
    T_k3_approx = hlpr.calc_selfdisctance_approx(k1, T)

    print(ST_k3)
    print(S_k3_approx)
    print(T_k3_approx)
    print(S_k3_approx*T_k3_approx)
    print(math.sqrt(S_k3_approx * T_k3_approx))

    k3_corr = ST_k3/(math.sqrt(S_k3_approx*T_k3_approx))"""
    k3_corr = ST_k3/(math.sqrt(S_k3 * T_k3))

    return k3_corr


def protein_string_kernel(data, features, beta=.2, k_max=10):

    sm2 = get_sm2()
    k1 = calc_k1_matrix(sm2, beta)

    n = int(len(data))
    m = int(len(features))

    kernel_matrix = np.zeros((n, m))

    for i in range(0, n):
        for j in range(0, m):
            # print(i,j)
            if i == j:
                kernel_matrix[i][j] = 1
                continue
            kernel_matrix[i][j] = calc_k3_correlation(k1, data[i], features[j])

    return kernel_matrix

if __name__ == "__main__":
    protein_string_kernel(None, .01)
