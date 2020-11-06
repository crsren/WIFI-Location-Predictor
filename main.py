import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from collections import deque
import numpy as np
from testing import *
from training import *
from node import *

def plot_graph(root, x1, x2, y1, y2, gap, ax):
    q1 = [(root, x1, x2, y1, y2)]
    while len(q1) > 0:
        q2 = q1.pop(0)
        node, x1, x2, y1, y2 = q2[0], q2[1], q2[2], q2[3], q2[4]
        a, v, p = node.attribute, node.value, node.pruned
        text = '['+str(a)+']@'+str(v)  # +'->'+str(p)
        c = x1+(x2-x1)/2.0
        d = (c-x1)/2.0

        if node.left is not None:
            q1.append((node.left, x1, c, y1, y2-gap))
            ax.annotate(text, xy=(c-d, y2-gap),
                        xytext=(c, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

        if node.right is not None:
            q1.append((node.right, c, x2, y1, y2-gap))
            ax.annotate(text, xy=(c+d, y2-gap),
                        xytext=(c, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

        if node.left is None and node.right is None:
            an1 = ax.annotate(node.leaf, xy=(c, y2), xycoords="data", va="bottom", ha="center",
                              bbox=dict(color="green", boxstyle="circle", fc="w"))

def max_depth(node):
    if node is None:
        return 0 ;

    else :

        # Compute the depth of each subtree
        lDepth = max_depth(node.left)
        rDepth = max_depth(node.right)

        # Use the larger one
        if (lDepth > rDepth):
            return lDepth+1
        else:
            return rDepth+1


def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")


def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")


def main():
    # np.set_printoptions(threshold=np.inf)
    clean_ds = loadClean()
    noisy_ds = loadNoisy()
    entire_ds = np.vstack((clean_ds, noisy_ds))
    #np.random.shuffle(clean_ds)
    np.random.shuffle(noisy_ds)
    #np.random.shuffle(entire_ds)
    #confusionAvg = prunedCrossValidate_confusion(entire_ds)
    #confusionAvg = crossValidate_confusion(entire_ds)
    #print("Average: ", confusionAvg)

    lol = np.split(noisy_ds, 20)
    mini_ds = lol[0]

    folds = np.split(mini_ds, 2)
    testSet = folds[0]
    trainingSet = folds[1]
    # # print(len(testSet))
    # # print(folds[0])
    # #trainingSet = np.concatenate(folds[1:])
    # print(testSet)
    # print("––––––––––––––––––")
    # print(trainingSet)

    root, depth, leafCount = decision_tree_learning(trainingSet)

    md = max_depth(root)

    fig, ax = plt.subplots(figsize=(1000, 10))
    gap = 1.0/depth
    plt.axis('off')
    plt.title("Unpruned Tree, depth:"+str(md), loc='left')
    plot_graph(root, 0.0, 1.0, 0.0, 1.0, gap, ax)
    #ax.set_title("Unpruned Tree, depth:"+str(md), align="left")
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.03)
    fig.subplots_adjust(left=0.03)
    fig.subplots_adjust(right=0.99)
    plt.show()


    print("unpruned depth:", md)

    #print(evaluate(testSet, root))

    print("Pruning!")
    # # Split into actual validation set later!!!
    root.perfectlyPruned(testSet, root)

    # avgAccuracy = crossValidate(ds)
    # print("average accuracy: ", avgAccuracy)
    #prune(root, testSet)

    nmd = max_depth(root)

    fig, ax = plt.subplots(figsize=(1000, 10))
    gap = 1.0/depth
    plt.axis('off')
    plt.title("Pruned Tree, depth:"+str(nmd), loc='left')
    plot_graph(root, 0.0, 1.0, 0.0, 1.0, gap, ax)
    #ax.set_title("Pruned Tree, depth:"+str(nmd), align="left")
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.03)
    fig.subplots_adjust(left=0.03)
    fig.subplots_adjust(right=0.99)
    plt.show()


    print("pruned depth:", nmd)

    return  # root, testSet


if __name__ == '__main__':
    main()
