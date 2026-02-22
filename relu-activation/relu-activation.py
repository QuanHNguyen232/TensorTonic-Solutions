import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.array(x)
    ans1 = np.where(x > 0, x, 0)
    ans2 = np.maximum(0, x)
    assert (ans1 == ans2).all()
    return ans1
