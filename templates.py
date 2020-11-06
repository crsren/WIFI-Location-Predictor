from main2 import *


def avgConfusionMatrix(ds=None):
    # print the clean dataset without prooning
    # runs on clean data set if no input given

    np.random.seed(100)
    # np.set_printoptions(threshold=np.inf)
    print("Loading and shuffeling clean dataset.")
    if(ds == None):
        ds = loadNoisy()

    np.random.shuffle(ds)

    print("Calculating the confusion matrices for 10-fold cross validation")
    confusionAvg = crossValidate_confusion(ds)
    print("Average confusion matrix: \n", confusionAvg)

# node.smartPrune(ds,root) → prunes a node only if it improves the trees performance
# node.perfectlyPruned(ds,root) → optimises the sub tree by recursively pruning every node below itself if it leads to performance improvements
