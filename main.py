import numpy as np
import networkx

clean_ds = np.loadtxt("wifi_db/clean_dataset.txt")
#noisy_ds = np.loadtxt("wifi_db/noisy_dataset.txt")
np.set_printoptions(threshold=np.inf)


class Node:

    def __init__ (self,  attribute, value=-1, left=None, right=None, leafornot=False):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leafornot = leafornot


def decision_tree_learning(ds, depth):

    left_ds = np.empty(shape=[0, 8])
    right_ds = np.empty(shape=[0, 8])

    firstLabel = ds[0][7]

    for currentLabel in ds:
        if(currentLabel[7] != firstLabel):
            #find_split → attribute index "i", decision value "n"
            c = find_split(ds)
            i = c[0]
            n = c[1]
            print ("Split on ", i, " by ", n)

            for row in ds:
                if(row[i] > n):
                    left_ds = np.vstack([left_ds,row])
                else:
                    right_ds = np.vstack([right_ds,row])

            print("LDS: ", left_ds.size/8, "RDS: ", right_ds.size/8)

            # recursion
            if(left_ds.size != 0): 
                lChild, lDepth = decision_tree_learning(left_ds, depth+1)
            if(right_ds.size != 0): 
                rChild, rDepth = decision_tree_learning(right_ds, depth+1)
            node = Node(i, n, lChild, rChild, False) # new node
            return  (node, max(lDepth, rDepth)) # return decision node

    print("Leaf:", firstLabel, depth, len(ds))
    return (Node(7, firstLabel, True), depth) # return leaf node

###trained_tree_node = {'attribute', 'value', 'left', 'right', leafornot (bool)}
# non leaf  -> Node(0, -87, Right, Left, False)
# leaf -> Node(7, 1)

###NOTES
# - stick with less than for decision
# - find_split returns {attribute index, int value} ?
#

def find_split(ds):

    # info_gain(ds[:,i])
    mx=0.0
    split_point=[0,0]
    for i in range(0,7):
        ds=ds[ds[:,i].argsort()]
        col = ds[:,i]
        rooms = ds[:,7]
    
    
        for j in range (int(min(col)), int(max(col))):
            for k in range(0,2000):
                if(col[k]>j):
                    Sleft = rooms[:k]
                    Sright = rooms[k:]
                    #ind=k
                    break

            gain = info_gain(ds[:,7],Sleft, Sright)
 
            if(gain>mx):
                mx=gain
                split_point=[i,j]


    ds=ds[ds[:,0].argsort()]

    return (split_point)

def info_gain(S,Sleft,Sright):
    return H(S) - remainder(Sleft,Sright)


def H(labels):
    p1=p2=p3=p4=0

    for i in labels:
        if i==1: p1 += 1
        elif i==2: p2 += 1
        elif i==3: p3 += 1
        elif i==4: p4 += 1

    if(p1!=0):
        p1 = p1*p1/len(labels)*np.log2(p1/len(labels))
    if(p2!=0):
        p2 = p2*p2/len(labels)*np.log2(p2/len(labels))
    if(p3!=0):
        p3 = p3*p3/len(labels)*np.log2(p3/len(labels))
    if(p4!=0):
        p4 = p4*p4/len(labels)*np.log2(p4/len(labels))

    return -(p1+p2+p3+p4)



def remainder(Sleft,Sright):
    return (len(Sleft)/(len(Sleft)+len(Sright)) * H(Sleft)) + (len(Sright)/(len(Sleft)+len(Sright)) * H(Sright))




def main():

    clean_ds = np.loadtxt("wifi_db/clean_dataset.txt")
    root, depth = decision_tree_learning(clean_ds, 0)
    print(depth)


    return


if __name__ == '__main__':
        main()
