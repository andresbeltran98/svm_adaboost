# Weighted LIBSVM:
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances
from libsvm_weights.svmutil import *
from libsvm_weights.commonutil import *
import math


class WeightedSVM:
    """
    Class to store a trained SVM classifier and its weight
    """
    def __init__(self, svm, weight=0):
        self.svm = svm
        self.weight = weight

    def predict(self, y, x):
        p_labels, p_acc_train, p_val_train = svm_predict(y, x, self.svm)
        return p_labels


class Ensemble:
    """
    Class to store a set of SVM classifiers
    """
    def __init__(self):
        self.classifiers = []

    def add(self, classifier):
        self.classifiers.append(classifier)

    def predict(self, y, x):
        """
        Predicts labels for novel examples
        :param y: true labels (for accuracy calculation)
        :param x: novel examples
        :return: a list of predicted labels and the accuracy metric
        """
        predictions = [0] * len(x)
        for cl in self.classifiers:
            p_labels = cl.predict(y, x)
            for i in range(len(predictions)):
                predictions[i] += (cl.weight * p_labels[i])

        # Calculate accuracy
        tp_plus_tn = 0
        for i in range(len(predictions)):
            predictions[i] = 1 if predictions[i] >= 0 else -1
            if y[i] == predictions[i]:
                tp_plus_tn += 1

        accuracy = tp_plus_tn / len(y)

        return predictions, (accuracy * 100)


def train_adaboost(weights, y, x, num_iter):
    """
    Trains and returns an AdaBoost ensemble
    :param weights: Initial weights
    :param y: true labels
    :param x: examples
    :param num_iter: number of iterations
    :return: an AdaBoost ensemble
    """
    ensemble = Ensemble()

    for iter in range(num_iter):

        print('Iteration', iter+1)

        # Train learner
        svm = svm_train(weights, y, x, '-t 0 -q')
        classifier = WeightedSVM(svm)
        p_labels, _, _ = svm_predict(y, x, svm)

        # Calculate weighted training error
        tr_error = 0
        for i in range(len(y)):
            if p_labels[i] != y[i]:
                tr_error += weights[i]

        # Set weight of this classifier
        classifier.weight = classifier_weight(tr_error)

        # Add classifier to ensemble
        ensemble.add(classifier)

        # Stopping conditions
        if tr_error == 0 or tr_error >= 0.5:
            break

        # Get normalization factor
        weights_sum = 0
        for i in range(len(weights)):
            weights_sum += weights[i] * math.exp(-1 * classifier.weight * y[i] * p_labels[i])

        # Update weights
        for i in range(len(weights)):
            weights[i] = (weights[i] * math.exp(-1 * classifier.weight * y[i] * p_labels[i])) / weights_sum

    return ensemble


def classifier_weight(tr_err):
    """
    Calculates the weight for a single AdaBoost classifier based on its training error
    :param tr_err: training error
    :return: weight for classifier
    """
    if tr_err == 0:
        return float(50000.0)

    return 0.5 * math.log((1-tr_err)/tr_err)


def run(K):
    y, x = svm_read_problem('DogsVsCats.train')

    init_weight = 1/len(y)
    w = [init_weight] * len(y)

    print('Training AdaBoost K =', K, '...')
    ensemble = train_adaboost(w, y, x, K)

    print('Testing...')
    y_test, x_test = svm_read_problem('DogsVsCats.test')
    p_pred, accuracy = ensemble.predict(y_test, x_test)
    print('K = ', K, 'Accuracy: ', accuracy)


if __name__ == "__main__":
    run(10)
    run(20)

