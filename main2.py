import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from collections import deque


def decision_tree_learning(ds, depth=0, leafCount=0):

    left_ds = np.empty(shape=[0, 8])
    right_ds = np.empty(shape=[0, 8])

    firstLabel = ds[0][7]

    for currentLabel in ds:
        if(currentLabel[7] != firstLabel):
            # find_split → attribute index "i", decision value "n"
            i, n = find_split(ds)
            #print("Split on ", i, " by ", n)

            for row in ds:
                if(row[i] > n):
                    right_ds = np.vstack([right_ds, row])
                else:
                    left_ds = np.vstack([left_ds, row])

            #print("LDS: ", left_ds.size/8, "RDS: ", right_ds.size/8)

            # recursivly split into subsets
            if(left_ds.size != 0):
                lChild, lDepth, leafCount = decision_tree_learning(
                    left_ds, depth+1, leafCount)
            if(right_ds.size != 0):
                rChild, rDepth, leafCount = decision_tree_learning(
                    right_ds, depth+1, leafCount)

            # return DECISION NODE
            return Node(i, n, ds, 0, lChild, rChild), max(lDepth, rDepth), leafCount

    #print("------ Leaf:", firstLabel, depth, len(ds))
    leafCount += 1
    # return LEAF NODE
    return Node(7, None, ds, firstLabel), depth, leafCount


def find_split(ds):

    maxGain = 0.0

    for i in range(0, 7):
        ds = ds[ds[:, i].argsort()]
        col = ds[:, i]
        rooms = ds[:, 7]

        # unique attributes in column i ordered in increasing order
        uniqueCol = np.unique(col)
        if(len(uniqueCol) == 1):
            continue
        for j in uniqueCol:
            for k in range(0, len(col)):
                if(col[k] > j):
                    Sleft = rooms[:k]
                    Sright = rooms[k:]
                    break

            gain = info_gain(ds[:, 7], Sleft, Sright)

            if(gain > maxGain):
                maxGain = gain
                attribute = i
                val = j

    return attribute, val


def info_gain(S, Sleft, Sright):
    return H(S) - remainder(Sleft, Sright)


def H(labels):
    p1 = p2 = p3 = p4 = 0

    for i in labels:
        if i == 1:
            p1 += 1
        elif i == 2:
            p2 += 1
        elif i == 3:
            p3 += 1
        elif i == 4:
            p4 += 1

    if(p1 != 0):
        p1 = p1*p1/len(labels)*np.log2(p1/len(labels))
    if(p2 != 0):
        p2 = p2*p2/len(labels)*np.log2(p2/len(labels))
    if(p3 != 0):
        p3 = p3*p3/len(labels)*np.log2(p3/len(labels))
    if(p4 != 0):
        p4 = p4*p4/len(labels)*np.log2(p4/len(labels))

    return -(p1+p2+p3+p4)


def remainder(Sleft, Sright):
    return (len(Sleft)/(len(Sleft)+len(Sright)) * H(Sleft)) + (len(Sright)/(len(Sleft)+len(Sright)) * H(Sright))


#from matplotlib import pyplot as plt
node = dict(boxstyle="square", fc="w")
connection = dict(arrowstyle="-")


class Node:
    # if leaf = 0 this is not a leaf node
    def __init__(self,  attribute, value, dataset, leaf=0, left=None, right=None, pruned=False):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf
        self.dataset = dataset
        self.pruned = pruned

    # def tree_copy():
    #     if(self.left != None):
    #         left = self.left.tree_copy()
    #     if(self.right != None):
    #         right = self.right.tree_copy()
    #     return Node(attribute, value, dataset, left, right)

    def perfectlyPruned(self, valset, root):
        # try pruning every node from the bottom up
        # if smartPrune returned false, no pruning above this one needs to be attempted
        if(self.leaf != 0):
            return True  # nothing to prune

        if(self.left != None):
            # prune left subtree as much as possible
            leftIsLeaf = self.left.perfectlyPruned(valset, root)
        if(self.right != None):
            # prune right subtree as much as possible
            rightIsLeaf = self.right.perfectlyPruned(valset, root)

        if(leftIsLeaf and rightIsLeaf):
            # both children had been turned into leaf nodes, try if pruning this one optimzes the tree
            return self.smartPrune(valset, root)
        else:
            return False  # At least one child wasn't pruned

    # returns True if pruning resulted in a more accuracte result, wherefore pruning was carried out
    # return False if the pruned tree was less accurate, wherefore the pruned was reversed
    def smartPrune(self, valset, root):
        self.pruned = True
        if(self.leaf):
            # nothing to prune (in case called outside of perfectly pruned)
            return True
        else:
            pre_accuracy = evaluate(valset, root)

            intLabels = self.dataset[:, 7].astype(np.int)
            self.leaf = np.argmax(np.bincount(intLabels))

            post_accuracy = evaluate(valset, root)

            print("pre: ", pre_accuracy, "post: ", post_accuracy)

            # prune even if same accuracy since more efficient → >=
            if(post_accuracy >= pre_accuracy):
                print("Pruned ", len(self.dataset))
                return True
            else:
                self.leaf = 0
                print("Didn't prune ", len(self.dataset))
                return False

    # function to draw tree from this node recursively
    # def plot(self, width, depth):
        # a = plt.figure(1,figsize=(8,8))
        # a.clf()
        # draw this node
        # if not a leaf node, draw connections to child nodes
        # left.plot()
        # right.plot()


def predict(node, signals):  # function to predict

    if(node.leaf):
        return node.leaf

    if(signals[node.attribute] > node.value):
        # print(node.attribute,": ", signals[node.attribute], " > ", node.value)
        return predict(node.right, signals)
    else:
        # print(node.attribute,": ", signals[node.attribute], " < ", node.value)
        return predict(node.left, signals)


def evaluate(ds, tree):  # return accuracy
    correct = 0

    for dp in ds:
        # print(dp)
        # print("Actual: ", dp[7], " | Predicted: ", predict(tree, dp))
        if(predict(tree, dp) == dp[7]):
            correct += 1

    return correct / len(ds)


def confusionMatrix(ds, tree):  # returns confusion matrix
    confusion = np.zeros((4, 4))  # actual, predicted

    for dp in ds:
        predicted = predict(tree, dp)
        #print("(CM) Actual: ", int(dp[7]), " | Predicted: ", predicted)
        confusion[int(dp[7])-1][int(predicted)-1] += 1

    return confusion

# def plotCM(cm):
    # TODO


# return an array of precision for each room, takes in a cunfussion matrix
def precisionmatrix(matrix):
    precision = np.zeros(4)
    for i in range(0, 4):
        tp = matrix[i][i]
        fp = matrix[i][0] + matrix[i][1] + matrix[i][2] + matrix[i][3]
        precision[i] = tp/(fp)

    return precision


def recallmatrix(matrix):  # return an array of recalls for each room takes in a cunfussion matrix

    recall = np.zeros(4)
    for i in range(0, 4):
        tp = matrix[i][i]
        fn = matrix[0][i] + matrix[1][i] + matrix[2][i] + matrix[3][i]
        recall[i] = tp/(fn)

    return recall


# takes in recall and precision array for each room and returns f1scores for each room
def f1score(recall, precision):
    f1score = np.zeros(4)
    for i in range(0, 4):
        f1score[i] = 2*((precision[i] * recall[i])/(precision[i] + recall[i]))

    return f1score


def crossValidate(ds, k=10):
    np.random.shuffle(ds)  # just in case this hasn't been done before
    folds = np.split(ds, k)
    accuracy = 0

    for i in range(0, k):
        # use i as test set and !i as training set
        testSet = folds[i]
        trainingSet = np.concatenate(
            list(folds[j] for j in range(0, k) if j != i))

        root, depth, leafCount = decision_tree_learning(trainingSet)

        accuracy += evaluate(testSet, root)

    return accuracy/k

    #training_ds= ds[:n,:]


def crossValidate_confusion(ds, k=10):
    np.random.shuffle(ds)  # just in case this hasn't been done before
    folds = np.split(ds, k)
    confusionTotal = np.zeros((4, 4))

    for i in range(0, k):
        # use i as test set and !i as training set
        testSet = folds[i]
        trainingSet = np.concatenate(
            list(folds[j] for j in range(0, k) if j != i))

        root, depth, leafCount = decision_tree_learning(trainingSet)

        confusion = confusionMatrix(testSet, root)
        print(i, ":", confusion)
        confusionTotal = np.add(confusionTotal, confusion)

    return confusionTotal/k


def prunedCrossValidate_confusion(ds, k=10):
    np.random.shuffle(ds)  # just in case this hasn't been done before
    folds = np.split(ds, k)
    confusionTotal = np.zeros((4, 4))

    for i in range(0, k):
        # use i as test set and !i as training set
        testSet = folds[i]
        trainingSet = np.concatenate(
            list(folds[j] for j in range(0, k) if j != i))

        root, depth, leafCount = decision_tree_learning(trainingSet)
        temp = confusionMatrix(testSet, root)

        root.perfectlyPruned(testSet, root)

        confusion = confusionMatrix(testSet, root)
        print(i, ":", confusion)
        confusionTotal = np.add(confusionTotal, confusion)

    return confusionTotal/k


def plot_graph(root, x1, x2, y1, y2, gap, ax):
    #q1 = deque([(root, x1, x2, y1, y2)])
    q1 = [(root, x1, x2, y1, y2)]
    while len(q1) > 0:
        q2 = q1.pop(0)
        node = q2[0]
        x1 = q2[1]
        x2 = q2[2]
        y1 = q2[3]
        y2 = q2[4]
        a = node.attribute
        v = node.value
        p = node.pruned
        text = '['+str(a)+']@'+str(v)  # +'->'+str(p)

        center = x1+(x2-x1)/2.0
        d = (center-x1)/2.0

        if node.left is not None:
            q1.append((node.left, x1, center, y1, y2-gap))
            ax.annotate(text, xy=(center-d, y2-gap),
                        xytext=(center, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

        if node.right is not None:
            q1.append((node.right, center, x2, y1, y2-gap))
            ax.annotate(text, xy=(center+d, y2-gap),
                        xytext=(center, y2), arrowprops=dict(arrowstyle="-"), bbox=dict(boxstyle="round", fc="w"))

        if node.left is None and node.right is None:
            an1 = ax.annotate(node.leaf, xy=(center, y2), xycoords="data", va="bottom", ha="center",
                              bbox=dict(color="green", boxstyle="circle", fc="w"))

def max_depth(node):
    if node is None:
        return 0 ;

    else :

        # Compute the depth of each subtree
        lDepth = max_depth(node.left) 
        rDepth = max_depth(node.right)

        # Use the larger one
        if (lDepth > rDepth):
            return lDepth+1
        else:
            return rDepth+1


def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")


def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")


def main():
    # np.set_printoptions(threshold=np.inf)
    clean_ds = loadClean()
    noisy_ds = loadNoisy()
    entire_ds = np.vstack((clean_ds, noisy_ds))
    np.random.shuffle(entire_ds)
    #confusionAvg = prunedCrossValidate_confusion(entire_ds)
    confusionAvg = crossValidate_confusion(entire_ds)
    print("Average: ", confusionAvg)

    lol = np.split(ds, 40)
    mini_ds = lol[0]

    folds = np.split(mini_ds, 2)
    testSet = folds[0]
    trainingSet = folds[1]
    # # print(len(testSet))
    # # print(folds[0])
    # #trainingSet = np.concatenate(folds[1:])
    # print(testSet)
    # print("––––––––––––––––––")
    # print(trainingSet)

    root, depth, leafCount = decision_tree_learning(trainingSet)

    fig, ax = plt.subplots(figsize=(1000, 10))
    gap = 1.0/depth
    plot_graph(root, 0.0, 1.0, 0.0, 1.0, gap, ax)
    fig.subplots_adjust(top=0.98)
    fig.subplots_adjust(bottom=0.03)
    fig.subplots_adjust(left=0.03)
    fig.subplots_adjust(right=0.99)
    plt.show()

    md = max_depth(root)
    print("unpruned depth:", md)

    #print(evaluate(testSet, root))

    print("Pruning!")
    # # Split into actual validation set later!!!
    root.perfectlyPruned(testSet, root)

    # avgAccuracy = crossValidate(ds)
    # print("average accuracy: ", avgAccuracy)
    #prune(root, testSet)

    fig, ax = plt.subplots(figsize=(1000, 10))
    gap = 1.0/depth
    plot_graph(root, 0.0, 1.0, 0.0, 1.0, gap, ax)
    fig.subplots_adjust(top=0.98)
    fig.subplots_adjust(bottom=0.03)
    fig.subplots_adjust(left=0.03)
    fig.subplots_adjust(right=0.99)
    plt.show()

    md = max_depth(root)
    print("pruned depth:", md)

    return  # root, testSet


if __name__ == '__main__':
    main()
