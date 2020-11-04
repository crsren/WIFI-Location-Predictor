import numpy as np
import node as Node

import main




def decision_tree_learning(ds, depth):

    left_ds = []
    right_ds = []
    firstLabel = ds[0][7]

    for currentLabel in ds:
        if(currentLabel[7] != firstLabel):
            #find_split → attribute index "i", decision value "n"
            i, n = find_split(ds)

            for row in ds:
                if(row[i] < n):
                    left_ds.append(row)
                else:
                    right_ds.append(row)

            # recursion
            lChild, lDepth = decision_tree_learning(left_ds, depth+1)
            rChild, rDepth = decision_tree_learning(right_ds, depth+1)
            node = Node(i, n, ds, lChild, rChild, False) #new node
            return  (node, max(lDepth, rDepth)) # return decision node

    return (Node(7, firstLabel, ds, None, None, True), depth) # return leaf node




clean_ds = np.loadtxt("wifi_db/clean_dataset.txt")

###trained_tree_node = {'attribute', 'value', 'left', 'right', leafornot (bool)}
# non leaf  -> Node(0, -87, Right, Left, False)
# leaf -> Node(7, 1)

root, depth= decision_tree_learning(clean_ds, 0)
