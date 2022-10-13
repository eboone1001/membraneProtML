# This is the implementation of the actual Support Vector Machine meant to use the Kernel implemented in
# ProtSeqKernel.py.
# Author: Eric Boone
# Date: 10/5/2022

import ProtSeqKernel
from sklearn import svm
import pandas as pd
import numpy as np
import time
import AACount
import matplotlib.pyplot as plt



def read_data(filename):

    data_matrix = pd.read_csv(filename, sep=',', header=None)
    return data_matrix.iloc[:, 1], data_matrix.iloc[:, 0]

if __name__ == "__main__":
    prot_class_svm = svm.SVC(kernel="precomputed")

    train_data, train_labels = read_data("rawdata/short_data_train.csv")
    test_data, test_labels = read_data("rawdata/short_data_test.csv")

    # Train and test using the kernel implemented in ProtSeqKernel.py. This is the important one.

    train_start = time.time()
    train_gram_mat = ProtSeqKernel.protein_string_kernel(train_data, train_data)
    print("Training Matrix time:", time.time() - train_start)
    print(train_gram_mat)
    prot_class_svm.fit(train_gram_mat, train_labels)

    test_start = time.time()
    test_gram_mat = ProtSeqKernel.protein_string_kernel(test_data, train_data)
    print(test_gram_mat)
    plt.imshow(train_gram_mat, test_gram_mat, cmap='hot', interpolation='nearest')
    plt.show()
    print("Test Matrix time:", time.time() - test_start)
    print(np.array(test_labels))
    kernel_results = prot_class_svm.predict(test_gram_mat)
    print(kernel_results)

    """ This is another SVC that uses the amino-acid counts of the sequence.  It should act as a baseline, because 
    hopefully the above kernel should perform better than just counting the number of amino-acids."""

    # TODO: currently this baseline uses a linear classifier, which will not work
    prot_class_linsvm = svm.SVC()

    training_matrix = AACount.get_aa_feature_matrix(train_data)
    prot_class_linsvm.fit(training_matrix, train_labels)

    test_matrix = AACount.get_aa_feature_matrix(test_data)
    aacount_results = prot_class_linsvm.predict(test_matrix)
    print(np.array(test_labels))
    print(aacount_results)

    results = [kernel_results, aacount_results]

    tot_correct = [0,0]
    for j in range(2):
        for i in range(0, len(results)):
            if results[j][i] == np.array(test_labels)[i]:
                tot_correct[j] += 1

    print("Kernel accuracy:" + str(tot_correct[0]/len(results) * 100) + "%")
    print("Kernel accuracy:" + str(tot_correct[1]/len(results) * 100) + "%")
