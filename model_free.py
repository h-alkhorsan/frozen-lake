import numpy as np
from numpy.core.fromnumeric import argmax
from frozen_lake import *

def randomAction(random_state, average_r):
    actions = np.array(np.argwhere(average_r == np.amax(average_r))).flatten()
    return random_state.choice(actions, 1)[0]  

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  
    eta = np.linspace(eta, 0, max_episodes)  
    epsilon = np.linspace(epsilon, 0, max_episodes) 

    q = np.zeros((env.n_states, env.n_actions)) 
    t = 0
    for i in range(max_episodes):
        s = env.reset() 

        if(t < env.n_actions): 
            a = t 
        else:
            _a = randomAction(random_state, q[s])

            if(random_state.random(1) < epsilon[i]):
                a = random_state.choice(range(env.n_actions))
            else:
                a = _a 
        t += 1

        done = False
        while not done: 
            state, R, done = env.step(a)

            if(t < env.n_actions): 
                action = t 
            else:
                _a = randomAction(random_state, q[state])

                if(random_state.random(1) < epsilon[i]):
                    action = random_state.choice(range(env.n_actions))  
                else:
                    action = _a  
            t += 1

            q[s,a] += eta[i] * (R + gamma * q[state, action] - q[s,a])
            s = state
            a = action

    policy = q.argmax(axis=1) 
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  
    eta = np.linspace(eta, 0, max_episodes) 
    epsilon = np.linspace(epsilon, 0, max_episodes)  

    q = np.zeros((env.n_states, env.n_actions)) 
    t = 0
    for i in range(max_episodes):
        s = env.reset() 
        done = False
        while(not done): 

            if(t < env.n_actions):  
                a = t  
            else:
                _a = randomAction(random_state, q[s])

                if(random_state.random(1) < epsilon[i]):
                    a = random_state.choice(range(env.n_actions))  
                else:
                    a = _a 
            t += 1

            state, r, done = env.step(a) 

            q_max = max(q[state])  
            q[s,a] += eta[i] * (r + gamma * q_max - q[s,a])
            s = state
    policy = q.argmax(axis=1) 
    value = q.max(axis=1)

    return policy, value

################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        _a = np.argmax(q_values)
        A[_a] += (1.0 - epsilon)
        return A
    return policy_fn


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)

        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            q_max = max(q)
            best = [a for a in range(env.n_actions) if np.allclose(q_max, q[a])]
            a = random_state.choice(best)

        done = False
        while not done:
            _features, R, done = env.step(a)

            _q = _features.dot(theta)

            if random_state.rand() < epsilon[i]:
                next_action = random_state.choice(env.n_actions)
            else:
                q_max = max(_q)
                best = [na for na in range(env.n_actions) if np.allclose(q_max, _q[na])]
                next_action = random_state.choice(best)

            D = R + gamma*_q[next_action] - q[a]
            theta = theta + eta[i]*D*features[a]

            features = _features
            q = features.dot(theta)
            a = next_action

    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed) 
    eta = np.linspace(eta, 0, max_episodes) 
    epsilon = np.linspace(epsilon, 0, max_episodes) 

    theta = np.zeros(env.n_features) 
    t = 0
    for i in range(max_episodes):
        features = env.reset() 
        q = features.dot(theta) 

        done = False
        while not done: 

            if t < env.n_actions: 
                a = t 
            else:
                _a = randomAction(random_state, q)

                if random_state.random(1) < epsilon[i]:
                    a = random_state.choice(range(env.n_actions))
                else:
                    a = _a
            t += 1

            _features, r, done = env.step(a)
            delta = r - q[a] 
            q = _features.dot(theta)
            delta += (gamma * max(q)) 
            theta += eta[i] * delta * features[a] 
            features = _features

    return theta

