import numpy as np
from __init__ import *

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from collections import deque


def plot_graph(root, xmin, xmax, ymin, ymax, gap, ax):
    queue = deque([(root, xmin, xmax, ymin, ymax)])
    while len(queue) > 0:
        q = queue.popleft()
        node = q[0]
        xmin = q[1]
        xmax = q[2]
        ymin = q[3]
        ymax = q[4]
        atri = node.attribute
        val = node.value
        text = '['+str(atri)+']:'+str(val)

        center = xmin+(xmax-xmin)/2.0
        d = (center-xmin)/2.0

        if node.left != None:
            queue.append((node.left, xmin, center, ymin, ymax-gap))
            ax.annotate(text, xy=(center-d, ymax-gap),
                        xytext=(center, ymax), arrowprops=dict(arrowstyle="->"),)

        if node.right != None:
            queue.append((node.right, center, xmax, ymin, ymax-gap))
            ax.annotate(text, xy=(center+d, ymax-gap),
                        xytext=(center, ymax), arrowprops=dict(arrowstyle="->"),)

        if node.left is None and node.right is None:
            an1 = ax.annotate(node.leaf, xy=(center, ymax), xycoords="data", va="bottom", ha="center",
                              bbox=dict(boxstyle="round", fc="w"))


def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")


def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")


def main():
    np.random.seed(100)
    # np.set_printoptions(threshold=np.inf)
    ds = loadNoisy()
    np.random.shuffle(ds)

    folds = np.split(ds, 2)
    testSet = folds[0]
    # print(len(testSet))
    # print(folds[0])
    #trainingSet = np.concatenate(folds[1:])
    trainingSet = folds[1]
    print(testSet)
    print("––––––––––––––––––")
    print(trainingSet)

    root, depth, leafCount = decision_tree_learning(trainingSet)

    print(evaluate(testSet, root))

    print("Pruning!")
    # Split into actual validation set later!!!
    root.perfectlyPruned(testSet, root)

    # avgAccuracy = crossValidate(ds)
    # print("average accuracy: ", avgAccuracy)
    #prune(root, testSet)

    fig, ax = plt.subplots(figsize=(18, 10))
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
