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
            print("Split on ", i, " by ", n)

            for row in ds:
                if(row[i] > n):
                    right_ds = np.vstack([right_ds, row])
                else:
                    left_ds = np.vstack([left_ds, row])

            #print("LDS: ", left_ds.size/8, "RDS: ", right_ds.size/8)

            # recursion
            if(left_ds.size != 0):
                lChild, lDepth, leafCount = decision_tree_learning(
                    left_ds, depth+1, leafCount)
            if(right_ds.size != 0):
                rChild, rDepth, leafCount = decision_tree_learning(
                    right_ds, depth+1, leafCount)

            # return decision node
            return Node(i, n, ds, 0, lChild, rChild), max(lDepth, rDepth), leafCount

    #print("------ Leaf:", firstLabel, depth, len(ds))
    leafCount += 1
    # return leaf node
    return Node(7, None, ds, firstLabel), depth, leafCount


def find_split(ds):

    mx = 0.0
    # Sleft = np.empty(shape=[0, 1])
    # Sright = np.empty(shape=[0, 1])

    for i in range(0, 7):
        ds = ds[ds[:, i].argsort()]
        col = ds[:, i]
        rooms = ds[:, 7]

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

            if(gain > mx):
                mx = gain
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
    def __init__(self,  attribute, value, dataset, leaf=0, left=None, right=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf
        self.dataset = dataset

    # def tree_copy():
    #     if(self.left != None):
    #         left = self.left.tree_copy()
    #     if(self.right != None):
    #         right = self.right.tree_copy()
    #     return Node(attribute, value, dataset, left, right)

    def perfectlyPruned(self, valset, root):
        # try pruning every node from the bottom up
        # if smartPrune returned false, no pruning above this one needs to be attempted
        if(self.leaf):
            return True  # nothing to prune

        if(self.left != None):
            # prune left subtree as much as possible
            leftIsLeaf = self.left.perfectlyPruned(valset, root)
        if(self.right != None):
            # prune right subtree as much as possible
            rightIsLeaf = self.left.perfectlyPruned(valset, root)

        if(leftIsLeaf and rightIsLeaf):
            # both children had been turned into leaf nodes, try if pruning this one
            return self.smartPrune(valset, root)
        else:
            return False  # At least one child wasn't pruned

    # returns True if pruning resulted in a more accuracte result, wherefore pruning was carried out
    # return False if the pruned tree was less accurate, wherefore the pruned was reversed
    def smartPrune(self, valset, root):
        if(self.leaf):
            # nothing to prune (in case called outside of perfectly pruned)
            return True
        else:
            # WE MIGHT WANNA USE CROSS VALIDATE INSTEAD
            pre_accuracy = evaluate(valset, root)

            intLabels = self.dataset[:, 7].astype(np.int)
            self.leaf = np.argmax(np.bincount(intLabels))

            # WE MIGHT WANNA USE CROSS VALIDATE INSTEAD
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

    return accuracy/float(k)

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

        confusionTotal = confusionTotal/float(k)
        print("Average: ", confusionTotal)
    return confusionTotal


def plot_graph(root, xmin, xmax, ymin, ymax, gap, ax):
    queue = deque([(root, xmin, xmax, ymin, ymax)])
    while len(queue) > 0:
        q = queue.popleft()
        node = q[0]
        xmin = q[1]
        xmax = q[2]
        ymin = q[3]
        ymax = q[4]
        atri = node.attribute
        val = node.value
        text = '['+str(atri)+']:'+str(val)

        center = xmin+(xmax-xmin)/2.0
        d = (center-xmin)/2.0

        if node.left != None:
            queue.append((node.left, xmin, center, ymin, ymax-gap))
            ax.annotate(text, xy=(center-d, ymax-gap),
                        xytext=(center, ymax), arrowprops=dict(arrowstyle="->"),)

        if node.right != None:
            queue.append((node.right, center, xmax, ymin, ymax-gap))
            ax.annotate(text, xy=(center+d, ymax-gap),
                        xytext=(center, ymax), arrowprops=dict(arrowstyle="->"),)

        if node.left is None and node.right is None:
            an1 = ax.annotate(node.leaf, xy=(center, ymax), xycoords="data", va="bottom", ha="center",
                              bbox=dict(boxstyle="round", fc="w"))


def loadClean():
    return np.loadtxt("wifi_db/clean_dataset.txt")


def loadNoisy():
    return np.loadtxt("wifi_db/noisy_dataset.txt")


def main():
    np.random.seed(100)
    # np.set_printoptions(threshold=np.inf)
    ds = loadClean()
    np.random.shuffle(ds)
    confusionAvg = crossValidate_confusion(ds)

    # folds = np.split(ds, 2)
    # testSet = folds[0]
    # # print(len(testSet))
    # # print(folds[0])
    # #trainingSet = np.concatenate(folds[1:])
    # trainingSet = folds[1]
    # print(testSet)
    # print("––––––––––––––––––")
    # print(trainingSet)

    #root, depth, leafCount = decision_tree_learning(trainingSet)

    #print(evaluate(testSet, root))

    # print("Pruning!")
    # # Split into actual validation set later!!!
    # root.perfectlyPruned(testSet, root)

    # avgAccuracy = crossValidate(ds)
    # print("average accuracy: ", avgAccuracy)
    #prune(root, testSet)

    # fig, ax = plt.subplots(figsize=(18, 10))
    # gap = 1.0/depth
    # plot_graph(root, 0.0, 1.0, 0.0, 1.0, gap, ax)
    # fig.subplots_adjust(top=0.98)
    # fig.subplots_adjust(bottom=0.03)
    # fig.subplots_adjust(left=0.03)
    # fig.subplots_adjust(right=0.99)
    # plt.show()

    return  # root, testSet


if __name__ == '__main__':
    main()
