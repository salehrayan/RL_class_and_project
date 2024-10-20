import numpy as np

def dynamic_programming(P, R, gamma, theta=1e-6):
    N = len(P)
    V = np.zeros(N)  # Initial value function
    delta = float('inf')
    
    while delta > theta:
        delta = 0
        V_new = np.copy(V)
        for s in range(N):
            V_new[s] = R[s] + gamma * np.sum(P[s, :] * V)
        delta = np.max(np.abs(V_new - V))
        V = V_new

    return V

P = np.array([[0.6, 0.4, 0.0],
              [0.4, 0.2, 0.4],
              [0.0, 0.4, 0.6]])

R = np.array([1, 0, 10])
gamma = 0.9

V = dynamic_programming(P, R, gamma)
print("Value:")
print(V)
