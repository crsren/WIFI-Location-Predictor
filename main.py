import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
from testing import *
from training import *
from node import *
import sys


def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")


def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")


def main(argv):

    try:
        ds = np.loadtxt(argv[0])
    except:
        print("Unable to open the file, using clean instead")
        ds = loadClean()

    np.random.shuffle(ds)

    folds = np.split(ds, 10)
    testSet = folds[0]
    trainingSet = np.concatenate(folds[1:])

    root, depth, leafCount = decision_tree_learning(trainingSet)
    pruned = False
    root.draw(pruned)
    root.perfectlyPruned(testSet, root)
    pruned = True
    root.draw(pruned)

    print("10 fold cross validated accuracy: ", crossValidate(ds))

    matrix = crossValidate_confusion(ds)
    matrixprunned = prunedCrossValidate_confusion(ds)

    print("confussion matrix non prunned: ", matrix)
    print("confussion matrix prunned: ", matrixprunned)


    return  # root, testSet


if __name__ == '__main__':
    main(sys.argv[1:])
