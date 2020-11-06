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

    # def tree_copy():
    #     if(self.left != None):
    #         left = self.left.tree_copy()
    #     if(self.right != None):
    #         right = self.right.tree_copy()
    #     return Node(attribute, value, dataset, left, right)

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
                return True
            else:
                self.leaf = 0
                print("Didn't prune ", len(self.dataset))
                return False

    # function to draw tree from this node recursively
    # def plot(self, width, depth):
        # a = plt.figure(1,figsize=(8,8))
        # a.clf()
        # draw this node
        # if not a leaf node, draw connections to child nodes
        # left.plot()
        # right.plot()
