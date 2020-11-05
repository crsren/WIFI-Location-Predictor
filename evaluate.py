import numpy as np
from __init__ import *


def predict(node, signals):  # function to predict

    if(node.leaf):
        return node.leaf

    if(signals[node.attribute] > node.value):
        # print(node.attribute,": ", signals[node.attribute], " > ", node.value)
        return predict(node.right, signals)
    else:
        # print(node.attribute,": ", signals[node.attribute], " < ", node.value)
        return predict(node.left, signals)


def evaluate(ds, tree):  # return accuracy
    correct = 0

    for dp in ds:
        print(dp)
        print("Actual: ", dp[7], " | Predicted: ", predict(tree, dp))
        if(predict(tree, dp) == dp[7]):
            correct += 1

    return correct / len(ds)


def confusionMatrix(ds, tree):  # returns confusion matrix
    confusion = np.zeros([4][4])  # actual, predicted

    for dp in ds:
        print("(CM) Actual: ", ds[7], " | Predicted: ", predict(tree, dp))
        confusion[ds[7]][predict(tree, dp)] += 1

    return confusion

# def plotCM(cm):
    # TODO


def crossValidate(ds, k=10):
    np.random.shuffle(ds)  # just in case this hasn't been done before
    folds = np.split(ds, k)
    accuracy = 0

    for i in range(0, k):
        # use i as test set and !i as training set
        testSet = folds[i]
        trainingSet = np.concatenate(
            list(folds[j] for j in range(0, k) if j != i))

        root, depth, leafCount = decision_tree_learning(trainingSet)

        accuracy += evaluate(testSet, root)

    return accuracy/k

    #training_ds= ds[:n,:]
