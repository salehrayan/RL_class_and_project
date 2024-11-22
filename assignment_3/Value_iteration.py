import numpy as np

def optimal_policy_via_bellman(rewards, transitions, gamma, epsilon=0.01, max_iterations=1000):
    n_states, n_actions = rewards.shape

    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    for iteration in range(max_iterations):
        delta = 0
        new_V = np.zeros(n_states)
        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * V)
            new_V[s] = np.max(q_values)
            policy[s] = np.argmax(q_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < epsilon:
            break

    return V, policy

rewards = np.array([
    [5, 10],
    [2, -1]
])

transitions = np.array([
    [[0.8, 0.2], [0.1, 0.9]],
    [[0.5, 0.5], [0.3, 0.7]]
])

gamma = 0.9
epsilon = 0.01
max_iterations = 1000

V, policy = optimal_policy_via_bellman(rewards, transitions, gamma, epsilon, max_iterations)

print("Optimal Value Function:")
print(V)
print("Optimal Policy:")
print(policy)
