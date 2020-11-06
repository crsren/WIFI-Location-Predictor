import numpy as np
from training import *

def predict(node, signals):  # function to predict

    if(node.leaf):
        return node.leaf

    if(signals[node.attribute] > node.value):
        return predict(node.right, signals)
    else:
        return predict(node.left, signals)


def evaluate(ds, tree):  # return accuracy
    correct = 0

    for dp in ds:
        if(predict(tree, dp) == dp[7]):
            correct += 1

    return correct / len(ds)


def confusionMatrix(ds, tree):  # returns confusion matrix
    confusion = np.zeros((4, 4))  # actual, predicted

    for dp in ds:
        predicted = predict(tree, dp)
        confusion[int(dp[7])-1][int(predicted)-1] += 1

    return confusion



# return an array of precision for each room, takes in a cunfussion matrix
def precisionmatrix(matrix):
    precision = np.zeros(4)
    for i in range(0, 4):
        tp = matrix[i][i]
        fp = matrix[i][0] + matrix[i][1] + matrix[i][2] + matrix[i][3]
        precision[i] = tp/(fp)

    return precision


def recallmatrix(matrix):  # return an array of recalls for each room takes in a cunfussion matrix

    recall = np.zeros(4)
    for i in range(0, 4):
        tp = matrix[i][i]
        fn = matrix[0][i] + matrix[1][i] + matrix[2][i] + matrix[3][i]
        recall[i] = tp/(fn)

    return recall


# takes in recall and precision array for each room and returns f1scores for each room
def f1score(recall, precision):
    f1score = np.zeros(4)
    for i in range(0, 4):
        f1score[i] = 2*((precision[i] * recall[i])/(precision[i] + recall[i]))

    return f1score


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



def crossValidate_confusion(ds, k=10):
    np.random.shuffle(ds)  # just in case this hasn't been done before
    folds = np.split(ds, k)
    confusionTotal = np.zeros((4, 4))

    for i in range(0, k):
        # use i as test set and !i as training set
        testSet = folds[i]
        trainingSet = np.concatenate(
            list(folds[j] for j in range(0, k) if j != i))

        root, depth, leafCount = decision_tree_learning(trainingSet)

        confusion = confusionMatrix(testSet, root)
        print(i, ":", confusion)
        confusionTotal = np.add(confusionTotal, confusion)

    return confusionTotal/k


def prunedCrossValidate_confusion(ds, k=10):
    np.random.shuffle(ds)  # just in case this hasn't been done before
    folds = np.split(ds, k)
    confusionTotal = np.zeros((4, 4))

    for i in range(0, k):
        # use i as test set and !i as training set
        testSet = folds[i]
        trainingSet = np.concatenate(
            list(folds[j] for j in range(0, k) if j != i))

        root, depth, leafCount = decision_tree_learning(trainingSet)
        temp = confusionMatrix(testSet, root)

        root.perfectlyPruned(testSet, root)

        confusion = confusionMatrix(testSet, root)
        print(i, ":", confusion)
        confusionTotal = np.add(confusionTotal, confusion)

    return confusionTotal/k
