import numpy as np

from node import Node
from build import *
#from prune import
from evaluate import predict


def main():

    clean_ds = np.loadtxt("wifi_db/clean_dataset.txt")
    noisy_ds = np.loadtxt("wifi_db/noisy_dataset.txt")
    np.set_printoptions(threshold=np.inf)
    
    root, depth, leafCount = decision_tree_learning(clean_ds, 0,0)
    print("Depth ",depth)
    print("leafCount ",leafCount)

    arr = np.array([-63,	-65,	-60,	-63,	-77,	-81,	-87])

    predict(root,arr)

    return root



if __name__ == '__main__':
        main()
