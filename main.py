import numpy as np
from node import *
from build import *
from evaluate import *

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from collections import deque


def plot_graph(root, x1, x2, y1, y2, gap, ax):
    #q1 = deque([(root, x1, x2, y1, y2)])
    q1 = [(root, x1, x2, y1, y2)]
    while len(q1) > 0:
        q2 = q1.pop(0)
        node = q2[0]
        x1 = q2[1]
        x2 = q2[2]
        y1 = q2[3]
        y2 = q2[4]
        a = node.attribute
        v = node.value
        text = '['+str(a)+']@'+str(v)

        center = x1+(x2-x1)/2.0
        d = (center-x1)/2.0

        if node.left is not None:
            q1.append((node.left, x1, center, y1, y2-gap))
            ax.annotate(text, xy=(center-d, y2-gap),
                        xytext=(center, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

        if node.right is not None:
            q1.append((node.right, center, x2, y1, y2-gap))
            ax.annotate(text, xy=(center+d, y2-gap),
                        xytext=(center, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

        if node.left is None and node.right is None:
            an1 = ax.annotate(node.leaf, xy=(center, y2), xycoords="data", va="bottom", ha="center",
                              bbox=dict(color="green", boxstyle="circle", fc="w"))


def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")


def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")


def main():
    np.random.seed(100)
    # np.set_printoptions(threshold=np.inf)
    ds = loadClean()
    np.random.shuffle(ds)

    lol = np.split(ds,40)
    mini_ds = lol[0]

    folds = np.split(mini_ds, 2)
    testSet = folds[0]
    # print(len(testSet))
    # print(folds[0])
    #trainingSet = np.concatenate(folds[1:])
    trainingSet = folds[1]
    print(testSet)
    print("––––––––––––––––––")
    print(trainingSet)

    confusionAvg = crossValidate_confusion(ds)

    # folds = np.split(ds, 2)
    # testSet = folds[0]
    # # print(len(testSet))
    # # print(folds[0])
    # #trainingSet = np.concatenate(folds[1:])
    # trainingSet = folds[1]
    # print(testSet)
    # print("––––––––––––––––––")
    # print(trainingSet)


    #root, depth, leafCount = decision_tree_learning(trainingSet)

    #print(evaluate(testSet, root))

    print("Pruning!")
    # Split into actual validation set later!!!
    root.perfectlyPruned(testSet, root)

    # avgAccuracy = crossValidate(ds)
    # print("average accuracy: ", avgAccuracy)
    #prune(root, testSet)

    fig, ax = plt.subplots(figsize=(1000, 10))
    gap = 1.0/depth
    plot_graph(root, 0.0, 1.0, 0.0, 1.0, gap, ax)
    fig.subplots_adjust(top=0.98)
    fig.subplots_adjust(bottom=0.03)
    fig.subplots_adjust(left=0.03)
    fig.subplots_adjust(right=0.99)
    plt.show()

    return root, testSet


if __name__ == '__main__':
    main()
