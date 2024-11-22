import numpy as np
import gym
from gym import spaces

# محیط دوبعدی 
class Simple2DEnv(gym.Env):
    def __init__(self):
        super(Simple2DEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # حرکت: بالا، پایین، چپ، راست
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32) # پنج در پنج
        self.state = np.array([0, 0])
        self.goal = np.array([4, 4])
    
    def reset(self):
        self.state = np.array([0, 0])
        return self.state
    
    def step(self, action):
        if action == 0: self.state[0] -= 1  # بالا
        elif action == 1: self.state[0] += 1  # پایین
        elif action == 2: self.state[1] -= 1  # چپ
        elif action == 3: self.state[1] += 1  # راست
        self.state = np.clip(self.state, 0, 4)  # محدودیت فضای محیط
        done = np.array_equal(self.state, self.goal)
        reward = 1 if done else -0.1  # ریوارد
        return self.state, reward, done, {}

# الگوریتم مونت کارلو فرست ویزیت
def monte_carlo_first_visit(env, episodes=500, gamma=0.9):
    Q = {} 
    returns = {}  
    for _ in range(episodes):
        state = env.reset()
        episode = []
        while True:
            action = env.action_space.sample()  # انتخاب تصادفی عمل
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            if (state[0], state[1], action) not in visited:
                visited.add((state[0], state[1], action))
                if (state[0], state[1], action) not in returns:
                    returns[(state[0], state[1], action)] = []
                returns[(state[0], state[1], action)].append(G)
                Q[(state[0], state[1], action)] = np.mean(returns[(state[0], state[1], action)])
    return Q

env = Simple2DEnv()
Q = monte_carlo_first_visit(env)
action_matrices = {i: np.zeros((5, 5)) for i in range(4)}

for (state_x, state_y, action), value in Q.items():
    if state_x < 5 and state_y < 5:  
        action_matrices[action][state_x, state_y] = value

action_names = ['Up', 'Down', 'Left', 'Right'] 
for action in range(4):
    print(f"\nAction: {action_names[action]}")
    print(action_matrices[action])
