import matplotlib.pyplot as plt

node=dict(boxstyle="square",fc="w")
connection=dict(arrowstyle="-")

class Node:

    def __init__ (self,  attribute, value, left=None, right=None, leaf=True):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf

    #function to draw tree from this node recursively
    #def plot(self, width, depth):
        # a = plt.figure(1,figsize=(8,8))
        # a.clf()
        #draw this node
        #if not a leaf node, draw connections to child nodes
        # left.plot() 
        # right.plot()
