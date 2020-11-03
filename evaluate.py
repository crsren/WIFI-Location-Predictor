from node import Node

# function to predict 
def predict(node, signals):

    if(node.leaf):
        print("Leaf:", node.value)
        return node.value

    
    if(signals[node.attribute] > node.value):
        print(node.attribute,": ", signals[node.attribute], " > ", node.value)
        predict(node.right, signals)
    else:
        print(node.attribute,": ", signals[node.attribute], " < ", node.value)
        predict(node.left, signals)

def crossValidate(ds, n=10):
    size = len(ds[:,0])/n
    #training_ds= ds[:n,:]

