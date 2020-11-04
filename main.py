import numpy as np

from node import Node
# from build import decision_tree_learning
#from prune import
from evaluate import *
from prune import *

def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")

def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")

def main():
    np.random.seed(100)
    #np.set_printoptions(threshold=np.inf)
    ds = loadClean()
    np.random.shuffle(ds)

    folds = np.split(ds,10)
    testSet = folds[0]
    #print(len(testSet))
    #print(folds[0])
    trainingSet = np.concatenate(folds[1:])
    print(trainingSet[len(trainingSet)-10:])
    print("––––––––––––––––––––––––––––––––––")
    print(ds[1990:])

    root, depth, leafCount = decision_tree_learning(trainingSet)

    print("––––––––––––––––––––––––––––––––––––––––")
    accuracy = evaluate(testSet,root)
    print(accuracy)

    avgAccuracy = crossValidate(ds)
    print("average accuracy: ", avgAccuracy)
    prune(root)

    return



if __name__ == '__main__':
        main()
