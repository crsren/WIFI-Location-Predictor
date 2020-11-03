
import numpy as np

from node import Node

def decision_tree_learning(ds, depth,leafCount):

    left_ds = np.empty(shape=[0, 8])
    right_ds = np.empty(shape=[0, 8])

    firstLabel = ds[0][7]

    for currentLabel in ds:
        if(currentLabel[7] != firstLabel):
            #find_split → attribute index "i", decision value "n"
            i,n = find_split(ds)
            print ("Split on ", i, " by ", n)

            for row in ds:
                if(row[i] > n):
                    left_ds = np.vstack([left_ds,row])
                else:
                    right_ds = np.vstack([right_ds,row])

            print("LDS: ", left_ds.size/8, "RDS: ", right_ds.size/8)

            # recursion
            if(left_ds.size != 0): 
                lChild, lDepth, leafCount = decision_tree_learning(left_ds, depth+1,leafCount)
            if(right_ds.size != 0): 
                rChild, rDepth, leafCount = decision_tree_learning(right_ds, depth+1,leafCount)
            node = Node(i, n, lChild, rChild,False) # new node
            return node, max(lDepth, rDepth), leafCount # return decision node

    print("------ Leaf:", firstLabel, depth, len(ds))
    leafCount += 1
    return Node(7,firstLabel), depth, leafCount # return leaf node


def find_split(ds):

    # info_gain(ds[:,i])
    mx=0.0
    split_point=[0,0]
    for i in range(0,7):
        ds=ds[ds[:,i].argsort()]
        col = ds[:,i]
        rooms = ds[:,7]
    
    
        for j in np.unique(col):
            for k in range(0,len(col)):
                if(col[k]>j):
                    Sleft = rooms[:k]
                    Sright = rooms[k:]
                    #ind=k
                    break

            gain = info_gain(ds[:,7],Sleft, Sright)
            if(gain>mx):
                mx=gain
                attribute = i
                val = j

    return attribute, val


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