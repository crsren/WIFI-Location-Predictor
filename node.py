import numpy as np
from __init__ import *
from evaluate import evaluate


#from matplotlib import pyplot as plt
node = dict(boxstyle="square", fc="w")
connection = dict(arrowstyle="-")


class Node:
    # if leaf = 0 this is not a leaf node
    def __init__(self,  attribute, value, dataset, leaf=0, left=None, right=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf
        self.dataset = dataset

    # def tree_copy():
    #     if(self.left != None):
    #         left = self.left.tree_copy()
    #     if(self.right != None):
    #         right = self.right.tree_copy()
    #     return Node(attribute, value, dataset, left, right)

    # returns True if pruning resulted in a more accuracte result, wherefore pruning was carried out
    # return False if the pruned tree was less accurate, wherefore the pruned was reversed
    def smartPrune(self, valse):
        if(self.leaf):
            return True  # nothing to prune
        else:
            # WE MIGHT WANNA USE CROSS VALIDATE INSTEAD
            pre_accuracy = evaluate(valset, self)

            intLabels = self.dataset[:, 7].astype(np.int)
            self.leaf = np.argmax(np.bincount(intLabels))

            # WE MIGHT WANNA USE CROSS VALIDATE INSTEAD
            post_accuracy = evaluate(valset, self)

            print("pre: ", pre_accuracy, "post: ", post_accuracy)

            # prune even if same accuracy since more efficient → >=
            if(post_accuracy >= pre_accuracy):
                return True
            else:
                self.leaf = 0
                return False

    # function to draw tree from this node recursively
    # def plot(self, width, depth):
        # a = plt.figure(1,figsize=(8,8))
        # a.clf()
        # draw this node
        # if not a leaf node, draw connections to child nodes
        # left.plot()
        # right.plot()
