import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    """
    X.shape = (B, feat)
    y.shape = (B,)
    """
    X, y = np.array(X), np.array(y)
    (B, feat) = X.shape
    w, b = np.zeros(feat), float(0)
    
    def predict(X, w, b):
        z = X @ w.T + b
        y_hat = _sigmoid(z)
        return y_hat
    
    def compute_loss(y, y_hat):
        """
        forward:
            L(y_hat) = - [ y*log(y_hat) + (1-y)*log(1-y_hat) ]
        backward:
            dL_dy_hat = - [ y/y_hat - (1-y)/(1-y_hat) ]
                      = - [ y*(1-y_hat) - y_hat*(1-y) / y_hat*(1-y_hat) ]
                      = - [ (y - y*y_hat - y_hat + y*y_hat) / y_hat*(1-y_hat) ]
                      = - [ (y - y_hat) / y_hat*(1-y_hat) ]
                      = (y_hat - y) / y_hat*(1-y_hat)
        """
        return np.mean(
            -1 * (y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        )
        
    def gradient(X, y, y_hat):
        B = X.shape[0]
        
        dL_dy_hat = (y_hat - y) / (y_hat*(1-y_hat))
        dy_hat_dz = y_hat*(1-y_hat) # (B,)
        dL_dz = dL_dy_hat * dy_hat_dz # chain rule
        
        grad_w = (X.T @ dL_dz) / B # (B, feat)
        grad_b = np.sum(dL_dz) / B
        return grad_w, grad_b
    
    def update(w, b, grad_w, grad_b):
        w -= lr * grad_w
        b -= lr * grad_b
        return w, b
        
    for _ in range(steps):
        y_hat = predict(X, w, b)
        loss = compute_loss(y, y_hat)
        grad_w, grad_b = gradient(X, y, y_hat)
        w, b = update(w, b, grad_w, grad_b)
    
    return w, b
        
    