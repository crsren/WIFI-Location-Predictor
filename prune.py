from node import Node
from evaluate import *

def prune(node):

    save_tree = node.tree_copy()
    pre_accuracy = evaluate(dataset, node)

    if (node.right.leafornot is False):
        prune(node.right)
    elif (node.left.leafornot is False):
        prune(node.left)
    else:
        print(node)

        if(node.left.ds.size() > node.right.ds.size()):
            node.value = node.left.value
            node.attribute = node.left.attribute
        else:
            node.value = node.right.value
            node.attribute = node.left.attribute

        node.left = None
        node.right = None

        post_accuracy = evaluate(dataset, node)

        if(post_accuracy >= pre_accuracy):
            #keep prune
            print("keep prune")
        else:
            #restore leaves
            print("restore leaves")


        #remove node (turn into leaf with side of higher number samples)
        #test new tree on validation set_printoption
        #use tree class
        #britney bitch
