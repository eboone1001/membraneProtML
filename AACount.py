# Amino Acid Count: This python file contains an implementation of function that determines whether a protein is
# an Alpha-helical or Beta-barrel protein for use as features in a classification algorithm

from sklearn import svm
import numpy as np
import membraneProtML_helper as hlpr

""" The purpose of these methods is to serve as a baseline for a ML feature selection.  It is meant to be a Naive 
solution so that we can compare our more complicated feature selection/kernel strategies, with the hope that they will
outperform this one. Calculating features is very simple, just count up the totals of each AA in the sequence and 
normalize.

NOTE: A binary classification of Alpha-helcies vs Beta-barrel transmembrane proteins works with 100% accuracy.  This 
shouldn't be shocking, as the AA makeup of a-helicies and b-sheets are quite different.
"""


def count_AA(sequence):

    aa_counts = np.zeros(26)
    for char in sequence:
        ind = hlpr.AA_letter_to_index[char]
        aa_counts[ind-1] += 1

    return aa_counts


def normalize_count(aa_counts):

    total = aa_counts.sum()
    aa_normalized_count = [count/total for count in aa_counts]

    return aa_normalized_count


def get_aa_feature_vect(sequence):
    return normalize_count(count_AA(sequence))


def get_aa_feature_matrix(sequences):
    feature_matrix = [get_aa_feature_vect(seq) for seq in sequences]
    return np.array(feature_matrix)

# For testing
if __name__ == "__main__":

    # TODO: currently this baseline uses a linear classifier, which will not work
    prot_class_linsvm = svm.SVC()

    train_data, train_labels = hlpr.read_data("rawdata/data_train.csv")
    test_data, test_labels = hlpr.read_data("rawdata/data_test.csv")

    training_matrix = get_aa_feature_matrix(train_data)
    prot_class_linsvm.fit(training_matrix, train_labels)

    test_matrix = get_aa_feature_matrix(test_data)
    aacount_results = prot_class_linsvm.predict(test_matrix)
    print(np.array(test_labels))
    print(aacount_results)

    tot_correct = 0
    for i in range(0, len(aacount_results)):
        if aacount_results[i] == np.array(test_labels)[i]:
            tot_correct += 1

    print("Kernel accuracy:" + str(tot_correct/len(aacount_results) * 100) + "%")