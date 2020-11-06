import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

node = dict(boxstyle="square", fc="w")
connection = dict(arrowstyle="-")


class Node:
    # if leaf = 0 this is not a leaf node


    def __init__(self,  attribute, value, dataset, leaf=0, left=None, right=None, pruned=False):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf
        self.dataset = dataset
        self.pruned = pruned


    def perfectlyPruned(self, valset, root):

        # try pruning every node from the bottom up
        # if smartPrune returned false, no pruning above this one needs to be attempted
        if(self.leaf != 0):
            return True  # nothing to prune

        if(self.left != None):
            # prune left subtree as much as possible
            leftIsLeaf = self.left.perfectlyPruned(valset, root)
        if(self.right != None):
            # prune right subtree as much as possible
            rightIsLeaf = self.right.perfectlyPruned(valset, root)

        if(leftIsLeaf and rightIsLeaf):
            # both children had been turned into leaf nodes, try if pruning this one optimzes the tree
            return self.smartPrune(valset, root)
        else:
            return False  # At least one child wasn't pruned

    # returns True if pruning resulted in a more accuracte result, wherefore pruning was carried out
    # return False if the pruned tree was less accurate, wherefore the pruned was reversed
    def smartPrune(self, valset, root):
        from testing import evaluate
        import numpy as np
        self.pruned = True
        if(self.leaf):
            # nothing to prune (in case called outside of perfectly pruned)
            return True
        else:
            pre_accuracy = evaluate(valset, root)

            intLabels = self.dataset[:, 7].astype(np.int)
            self.leaf = np.argmax(np.bincount(intLabels))

            post_accuracy = evaluate(valset, root)

            print("pre: ", pre_accuracy, "post: ", post_accuracy)

            # prune even if same accuracy since more efficient → >=
            if(post_accuracy >= pre_accuracy):
                print("Pruned ", len(self.dataset))
                self.left = None
                self.right = None
                self.leaf = 1
                return True
            else:
                self.leaf = 0
                print("Didn't prune ", len(self.dataset))
                return False


    def max_depth(self):
        # Compute the depth of each subtree
        if(self.left == None or self.right == None):
            return 0
        else:
            lDepth = self.left.max_depth()
            rDepth = self.right.max_depth()

        # Use the larger one
        if (lDepth > rDepth):
            return lDepth+1
        else:
            return rDepth+1


    def treeplot(self, x1, x2, y1, y2, space, ax):
        q1 = [(self, x1, x2, y1, y2)]
        while len(q1) > 0:
            q2 = q1.pop(0)
            node, x1, x2, y1, y2 = q2[0], q2[1], q2[2], q2[3], q2[4]
            c = x1+((x2-x1)/2)
            txt = str(node.attribute)+' @ '+str(node.value)  # +'->'+str(p)
            d = (c-x1)/2
            diff = y2-space

            if (node.left is not None):
                q1.append((node.left, x1, c, y1, diff))
                ax.annotate(txt, xy=(c-d, diff),
                            xytext=(c, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

            if (node.right is not None):
                q1.append((node.right, c, x2, y1, diff))
                ax.annotate(txt, xy=(c+d, diff),
                            xytext=(c, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

            if (node.left is None and node.right is None):
                ax.annotate(node.leaf, xy=(c, y2), xycoords="data", va="bottom", ha="center",
                                  bbox=dict(color="green", boxstyle="circle", fc="w"))

    def draw(self, pruned):
        fig, ax = plt.subplots(figsize=(1000, 10))
        depth = self.max_depth()
        space = 1.0/depth
        plt.axis('off')
        if(pruned is True):
            plt.title("Pruned Tree, depth:"+str(depth), loc='left')
        else:
            plt.title("Unpruned Tree, depth:"+str(depth), loc='left')
        self.treeplot(0.0, 1.0, 0.0, 1.0, space, ax)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(bottom=0.03)
        fig.subplots_adjust(left=0.03)
        fig.subplots_adjust(right=0.99)
        plt.show()
