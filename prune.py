from node import Node
from evaluate import *

def prune(node, depth):

    if (!node.right.leafornot):
        prune(node.right)
    else if (!node.left.leafornot):
        prune(node.left)
    else:
        print(node)

        if(node.left.ds.size() > node.right.ds.size()):
            node.value = node.left.value
        else:
            node.value = node.right.value

        node.left = None
        node.right = None

        #remove node (turn into leaf with side of higher number samples)
        #test new tree on validation set_printoption
        #use tree class
        #britney bitch
