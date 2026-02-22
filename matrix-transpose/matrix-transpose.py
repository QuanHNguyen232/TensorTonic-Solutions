import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    ans1 = np.transpose(A, axes=(1, 0))
    ans2 = A.T
    assert (ans1 == ans2).all()
    return ans1
