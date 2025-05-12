import numpy as np

xInp = np.linspace(0, 1, 100)
Z = np.sin(2 * np.pi * xInp)

def deterministicFitting():
    X = np.vander(xInp, 10, increasing=True)
    return np.linalg.solve(X.T @ X, X.T @ Z)

w = deterministicFitting()
x = xInp[2]
predicted = w[0] + w[1]*x + w[2]*x**2  # = sum(w[i] * x**i for i in range(3))
actual = Z[2]

print(f"At x = {x:.6f}:")
print(f"Predicted: {predicted:.6f}")
print(f"Actual:    {actual:.6f}")
print(f"Error:     {abs(predicted-actual):.6f}")
