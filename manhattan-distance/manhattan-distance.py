import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x, y = np.array(x), np.array(y)
    diff = np.absolute(x - y).flatten()
    return float(np.sum(diff))
    