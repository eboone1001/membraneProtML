# This is the implementation of the actual Support Vector Machine meant to use the Kernel implemented in
# ProtSeqKernel.py.
# Author: Eric Boone
# Date: 10/5/2022

import ProtSeqKernel
from sklearn import svm
import numpy as np
import time
import AACount
import matplotlib.pyplot as plt
import membraneProtML_helper as hlpr


if __name__ == "__main__":
    prot_class_svm = svm.SVC(kernel="precomputed")

    train_data, train_labels = hlpr.read_data("rawdata/short_data_train.csv")
    test_data, test_labels = hlpr.read_data("rawdata/short_data_test.csv")

    # Train and test using the kernel implemented in ProtSeqKernel.py.

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

    tot_correct = 0
    for i in range(0, len(kernel_results)):
        if kernel_results[i] == np.array(test_labels)[i]:
            tot_correct += 1

    print("Kernel accuracy:" + str(tot_correct[1]/len(kernel_results) * 100) + "%")
