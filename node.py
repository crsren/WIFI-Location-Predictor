class Node:

    def __init__ (self,  attribute, value, left=None, right=None, leafornot=True):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leafornot = leafornot

    #function to draw tree from this node