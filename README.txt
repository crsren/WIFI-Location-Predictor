# Coursework 1: Decision Trees

#### Training, tuning and testing a decision tree to determine the location of a mobile phone based on collected WIFI signal strengths.

### Training the tree

Pass the data set into the function:

e.g. python3 main.py <trainingSet>

Please replace <trainingSet> with the dataset you want the system to be trained on. Executing the main function without any arguments will default to the clean dataset given.

The tree is build using the "decision_tree_learning()" function that takes in the training set and returns root, depth, leafCount of the unpruned tree:
root, depth, width = decision_tree_learning(<trainingSet>)

### Pruning the tree

One can prune the tree by running the "perfectlyPruned()" method on the root of the tree, which takes in the testSet and the root of the tree:
root.perfectlyPruned(<testSet>,<root>)

### Drawing the tree

One can draw the tree by running the "draw()" method on the root of the tree, which takes in a boolean indicating if the tree has been pruned or not (True == pruned):
root.draw()

### Evaluating the tree

A tree can be evaluated on a <testSet> using the "evaluate()" function as follows, which returns the accuracy of the model:
accuracy = evaluate(<testSet>, <root>)

A more comprehensive evaluation can be achieved using "crossValidate()", which takes in the entire dataset as well as the desired number of folds and returns the average accuracy of the trained tree:
accuracy = crossValidate(<dataSet>, <folds>)
