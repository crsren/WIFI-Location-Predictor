import numpy as np

from node import Node
# from build import decision_tree_learning
#from prune import
from evaluate import *

def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")
    
def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")

def main():

    clean_ds = loadClean()
    np.set_printoptions(threshold=np.inf)
    
  #  root, depth, leafCount = decision_tree_learning(clean_ds)

    avgAccuracy = crossValidate(clean_ds)
    print("average accuracy: ", avgAccuracy)

    return



if __name__ == '__main__':
        main()
