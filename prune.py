from node import Node
from evaluate import *
import copy
from build import *



#save_tree = root.tree_copy()
#prunes = []

def prune(node, test_data):

    pre_accuracy = evaluate(test_data, node)

    if (node.left.leaf is False):
        prune(node.left, test_data)
    elif (node.right.leaf is False):
        prune(node.right, test_data)
    else:
        tempNode = copy.deepcopy(node)
        # tempNode = Node(node.attribute, node.value, node.left, node.right, node.leaf, node.dataset)
        # tmp.attribute = node.attribute
        # tmp.value = node.value
        # tmp.left = node.left
        # tmp.right = node.right
        # tmp.leaf = node.leaf
        # tmp.dataset = node.dataset

        if(len(node.left.dataset) > len(node.right.dataset)):
            node.value = node.left.value
            node.attribute = node.left.attribute
        else:
            node.value = node.right.value
            node.attribute = node.left.attribute

        node.left = None
        node.right = None
        node.leaf = True

        post_accuracy = evaluate(test_data, node)

        if(post_accuracy >= pre_accuracy):
            #keep prune
            print("keep prune")
            return node
        else:
            #restore leaves - need stack to store prunes prior to accuracy check?
            print("restore leaves")
            node.attribute = tempNode.attribute
            node.value = tempNode.value
            node.left = tempNode.left
            node.right = tempNode.right
            node.leaf = tempNode.leaf
            node.dataset = tempNode.dataset
            return node


        #remove node (turn into leaf with side of higher number samples)
        #test new tree on validation set_printoption
        #use tree class
        #britney bitch
