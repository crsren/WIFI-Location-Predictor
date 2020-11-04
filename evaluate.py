import numpy as np

from node import Node
from build import decision_tree_learning

# function to predict 
def predict(node, signals):

    if(node.leaf):
        print("Leaf:", node.value)
        return node.value

    
    if(signals[node.attribute] > node.value):
        # print(node.attribute,": ", signals[node.attribute], " > ", node.value)
        predict(node.right, signals)
    else:
        # print(node.attribute,": ", signals[node.attribute], " < ", node.value)
        predict(node.left, signals)

# return accuracy
def evaluate(ds, tree):
    correct = 0

    for dp in ds:
        print("Actual: ", ds[7], " | Predicted: ", predict(tree,dp))
        if(predict(tree, dp) == ds[7]):
            correct+=1
    
    return correct / len(ds)

# returns confusion matrix
def confusionMatrix(ds, tree):
    confusion = np.zeros([4][4]) #actual, predicted

    for dp in ds:
        print("(CM) Actual: ", ds[7], " | Predicted: ", predict(tree,dp))
        confusion[ds[7]][predict(tree,dp)]+=1
    
    return confusion

#def plotCM(cm):
    #TODO


def crossValidate(ds, k=10):
    np.random.shuffle(ds) # just in case this hasn't been done before
    folds = np.split(ds,k)
    accuracy = 0

    for i in range (0,k):
        #use i as test set and !i as training set
        testSet = folds[i]
        trainingSet = np.concatenate(list(folds[j] for j in range(0,k) if j != i))

        root, depth, leafCount = decision_tree_learning(trainingSet)

        accuracy += evaluate(testSet, root)

    return accuracy/k
        


    #training_ds= ds[:n,:]

