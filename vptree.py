import random
import numpy as np

class _VPNode():

    def __init__(self, datapoint):
        """
        a node in a vantage-point tree with possible left and right children
        """
        self.vpoint = datapoint
        self.radius = None
        self.left = None   # another node
        self.right = None  # another node

def _build_vptree(datapoints, distfun):
    """
    Inputs:
        - datapoints: all points that have not yet been put into the tree
        - distfun: function to compute the distance between 2 datapoints
    """
    # we can only make a node out of at least 1 element
    if not datapoints:
        return None
    node = _VPNode(datapoints[0])
    # if this was the last point, we're done
    if not datapoints[1:]:
        return node
    remaining_dps = datapoints[1:]
    # compute all distances of this point to the other points left
    distances = [distfun(datapoints[0], dp) for dp in remaining_dps]
    # compute the current datapoint's radius as the median of all distances
    node.radius = np.median(distances)
    # build left node based on elements inside the datapoint's radius
    # (would probably be faster with np boolean indexing)
    node.left = _build_vptree([dp for i, dp in enumerate(remaining_dps) if distances[i] <= node.radius], distfun)
    # and right node based on elements outside the datapoint's radius
    node.right = _build_vptree([dp for i, dp in enumerate(remaining_dps) if distances[i] > node.radius], distfun)
    return node


class VPTree():

    def __init__(self, datapoints, distfun):
        """
        Builds a vantage-point tree out of the given datapoints.

        Inputs:
            - datapoints: list of objects that should be put into the tree
            - distfun: a function that takes 2 of the datapoints and returns a float
                       indicating their distance - has to satisfy the triangle inequality!!
        """
        self.distfun = distfun
        random.shuffle(datapoints)
        # create the root node that contains all other nodes
        self.root = _build_vptree(datapoints, distfun)

    def find_knn(self, dp, k):
        """
        Find the k nearest neighbors of the given data point by checking the tree using its distance function
        """
        neighbors = []  # tuples of (point, distance)
        # distance to furthest nearest neighbor
        tau = np.inf
        # nodes that still have to be checked
        nodes_to_check = [self.root]
        # go through the tree to look for nearest neighbors
        while nodes_to_check:
            # is the current node a good candidate?
            node = nodes_to_check.pop(0)
            dist = self.distfun(dp, node.vpoint)
            if dist < tau or len(neighbors) < k:
                # keep as a neighbor
                neighbors.append((node.vpoint, dist))
                # sort by distance and if we have more than k, kick them out
                neighbors = sorted(neighbors, key=lambda x:x[1])[:min(len(neighbors),k)]
                # adapt tau
                tau = neighbors[-1][1]
            # see which other nodes should be explored
            # is the target node inside or outside the current node's ball? (where is the larger portion of possible nn?)
            if dist <= node.radius:
                # inside --> search left side first
                if node.left:
                    nodes_to_check.append(node.left)
                # are there some points outside the radius that could still be closer than tau?
                if node.right and dist + tau > node.radius:
                    nodes_to_check.append(node.right)
            else:
                # outside --> search right side first
                if node.right:
                    nodes_to_check.append(node.right)
                # are there some points inside the radius that could still be closer than tau?
                if node.left and dist - tau <= node.radius:
                    nodes_to_check.append(node.left)
        return neighbors
