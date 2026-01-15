# data.py
import numpy as np

def generate_batch(n=500, drift_strength=0.0):
    # Protejat vs Neprotejat (50/50 e mai stabil pentru demo)
    A = np.random.randint(0, 2, n) 
    
    # X corelat cu A
    X = np.random.randn(n, 2)
    X[:, 0] += A * 0.5 

    # Drift-ul afectează direct logica de decizie (discriminare "slow-creep")
    # drift_strength mare = penalizare mare pentru grupul A=0
    bias_against_0 = drift_strength 
    
    logits = X[:, 0] + 0.5 * X[:, 1] - bias_against_0 * (1 - A)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    return X, y, A