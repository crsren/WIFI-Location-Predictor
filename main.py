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


# def plot_graph(root, x1, x2, y1, y2, gap, ax):
#     q1 = [(root, x1, x2, y1, y2)]
#     while len(q1) > 0:
#         q2 = q1.pop(0)
#         node, x1, x2, y1, y2 = q2[0], q2[1], q2[2], q2[3], q2[4]
#         a, v, p = node.attribute, node.value, node.pruned
#         text = '['+str(a)+']@'+str(v)  # +'->'+str(p)
#         c = x1+(x2-x1)/2.0
#         d = (c-x1)/2.0
#
#         if node.left is not None:
#             q1.append((node.left, x1, c, y1, y2-gap))
#             ax.annotate(text, xy=(c-d, y2-gap),
#                         xytext=(c, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))
#
#         if node.right is not None:
#             q1.append((node.right, c, x2, y1, y2-gap))
#             ax.annotate(text, xy=(c+d, y2-gap),
#                         xytext=(c, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))
#
#         if node.left is None and node.right is None:
#             an1 = ax.annotate(node.leaf, xy=(c, y2), xycoords="data", va="bottom", ha="center",
#                               bbox=dict(color="green", boxstyle="circle", fc="w"))

# def max_depth(node):
#     if node is None:
#         return 0 ;
#
#     else :
#
#         # Compute the depth of each subtree
#         lDepth = max_depth(node.left)
#         rDepth = max_depth(node.right)
#
#         # Use the larger one
#         if (lDepth > rDepth):
#             return lDepth+1
#         else:
#             return rDepth+1


def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")


def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")


def main(argv):
    # np.set_printoptions(threshold=np.inf)
    #noisy_ds = loadNoisy()
    try:
        ds = np.loadtxt(argv[0])
    except:
        print("Unable to open the file, using clean instead")
        ds = loadClean()

    #entire_ds = np.vstack((ds, noisy_ds))


    #np.random.shuffle(entire_ds)
    #confusionAvg = prunedCrossValidate_confusion(entire_ds)
    #confusionAvg = crossValidate_confusion(ds)
    #print("Average: ", confusionAvg)
    np.random.shuffle(ds)

    folds = np.split(mini_ds, 2)
    testSet = folds[0]
    #trainingSet = np.concatenate(folds[1:])
    print(testSet)

    root, depth, leafCount = decision_tree_learning(trainingSet)
    pruned = False
    root.draw(pruned)
    root.perfectlyPruned(testSet, root)
    pruned = True
    root.draw(pruned)

    return  # root, testSet


if __name__ == '__main__':
    main(sys.argv[1:])
