import numpy as np
import math
clean_ds = np.loadtxt("wifi_db/clean_dataset.txt")
#noisy_ds = np.loadtxt("wifi_db/noisy_dataset.txt")

#print(noisy_ds)
#print(clean_ds)
#print(clean_ds[0])


###NOTES
# - stick with less than for decision
# - find_split returns {attribute index, int value} ?
#

def find_split(ds):
    ds[ds[:,i].argsort()]
    info_gain(ds[:,i])
    for i in range(0,6)
        col = ds[:,i]
        for j in range (min(col),max(col))
            for k in range(0,2000)
                if(col[k]>j)
                    push into Sright
                else
                    push into Sleft

            info_gain(ds[:,i],Sleft, Sright)

    return

def info_gain(S,Sleft,Sright):

    return H(S) - remainder(Sleft,Sright)


def H(labels):
    p1=0
    p2=0
    p3=0
    p4=0
    for i in labels
        if labels[i]==1
            p1 += 1
        elif labels[i]==2
            p2 += 1
        elif labels[i]==3
            p3 += 1
        else labels[i]==4
            p4 += 1

    p1 = p1/len(labels)
    p2 = p2/len(labels)
    p3 = p3/len(labels)
    p4 = p4/len(labels)

    return -(p1*math.log2(p1))+(p2*math.log2(p2))+(p3*math.log(p3))+(p4*math.log2(p4))



def remainder(Sleft,Sright):


    return (len(Sleft)/(len(Sleft)+len(Sright)) * H(Sleft)) + (len(Sright)/(len(Sleft)+len(Sright)) * H(Sright))
