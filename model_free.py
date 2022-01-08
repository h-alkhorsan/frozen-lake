import numpy as np
from numpy.core.fromnumeric import argmax
from frozen_lake import *

################ Tabular model-free algorithms ################

# epsilon-greedy exploration strategy
def epsilon_greedy(Q, epsilon, s):
    """
    Q: Q Table
    epsilon: exploration parameter
    s: state
    """
    # selects a random action with probability epsilon
    if np.random.random() <= epsilon:
            return np.random.randint(4)
    else:
            return np.random.choice(np.flatnonzero(Q[s, :] == Q[s, :].max()))

# SARSA Process
def sarsa(env, n_episodes,alpha , gamma, epsilon, seed=None): 
    """
    alpha: learning rate
    gamma: exploration parameter
    n_episodes: number of episodes
    """
    
    # initialize Q table
    
    Q = np.zeros((env.n_states, env.n_actions))

    step = 0

    alpha = np.linspace(alpha, 0, n_episodes)
    epsilon = np.linspace(epsilon, 0, n_episodes)


    # initialize processing bar
    # to record reward for each episode
    reward_array = np.zeros(n_episodes)
    for i in range(n_episodes):
            # initial state
            s = env.reset()
            # initial action
            a = epsilon_greedy(Q, epsilon[i], s)
            state_check = s
            step = 0

            done = False
            while not done:
                s_, reward, done = env.step(a)

                if ((done and reward == 0) or (done and reward == 1)):
                    #final state updated
                    s_ = state_check
                else:
                    state_check = s_

                a_ = epsilon_greedy(Q, epsilon[i], s_)


                Q[s, a] += alpha[i] * (reward + (gamma * Q[s_, a_]) - Q[s, a])
                # update processing bar
                if done:
                        reward_array[i] = reward
                        break
                
                step+=1
                s, a = s_, a_
    # show Q table
    print('Trained Q Table:')
    print(Q)
    # show average reward
    avg_reward = round(np.mean(reward_array), 4)
    print('Training Averaged reward per episode {}'.format(avg_reward))

    policy = Q.argmax(axis=1)

    print(policy)

    value = Q.max(axis=1)

    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    Q = np.zeros((env.n_states, env.n_actions))
    s = env.reset()

    state_check = s

    
    for i in range(max_episodes):
        s = env.reset()
        done = False
        
        while not done:
            a = epsilon_greedy(Q, epsilon[i], s)
            s_, reward, done = env.step(a)

            
            if ((done and reward == 0) or (done and reward == 1)):
                #final state updated
                s_ = state_check
            else:
                state_check = s_
            
            Q[s][a] = Q[s][a] + eta[i] * ((reward+gamma*Q[s_][np.random.choice(np.flatnonzero(Q[s_, :] == Q[s_, :].max()))]) - Q[s][a])
            s = s_
            
        
    policy = Q.argmax(axis=1)
    value = Q.max(axis=1)

    print(policy)
    print(value)
    print(Q)
        
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
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    estimator = LinearWrapper(env)
    done = False
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        policy = make_epsilon_greedy_policy(estimator, epsilon[i] * eta[i], env.n_actions)

        q = features.dot(theta)

        # TODO:
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    estimator = LinearWrapper(env)
    done = False

    for i in range(max_episodes):
        features = env.reset()
        policy = make_epsilon_greedy_policy(estimator, epsilon[i] * eta[i], env.n_actions)

        # while not done:



        
        # TODO:

    return theta    
