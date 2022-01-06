import numpy as np
from numpy.core.fromnumeric import argmax
from frozen_lake import *

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    def action_select(state, epsilon):
        action = 0
        greedy = np.random.uniform(0,1)

        if greedy < epsilon:
            action = np.random.randint(0,3)
        else:
            print('greedy')
            action = np.argmax(q[state,:])
        return action
    def update(state1, state2, reward, action1, action2, eta):
        predict = q[state1,action1]
        target = reward + gamma * q[state2, action2]
        q[state1,action1] =  q[state1,action1] + eta * (target - predict)

    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    reward = 0
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        t = 0
        s = env.reset()
        # TODO:
        action1 = action_select(s, epsilon[i])
        while t < env.max_steps:

            state2, reward, done = env.step(action1)

            env.render()
            action2 = action_select(state2, epsilon[i])
            update(s, state2, reward,action1,action2, eta[i])

            s = state2
            action1 = action2   

            t +=1
            reward +=1

            if done:
                break
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

        
    #Evaluating the performance
    print ("Performance : ", reward/max_episodes)
    
    #Visualizing the Q-matrix
    print(q)


    return policy, value



    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        
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
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:

    return theta    
