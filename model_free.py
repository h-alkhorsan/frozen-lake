import numpy as np
from numpy.core.fromnumeric import argmax
from model_based import policy_evaluation, policy_improvement
from frozen_lake import * 

################ Tabular model-free algorithms ################


def action_select(Q, epsilon, s):
    if np.random.random() <= epsilon:
            return np.random.randint(4)
    else:
            return np.random.choice(np.flatnonzero(Q[s, :] == Q[s, :].max()))


# SARSA Process
def sarsa(env, max_episodes, alpha , gamma, epsilon, seed=None):    
    Q = np.zeros((env.n_states, env.n_actions))


    alpha = np.linspace(alpha, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    for i in range(max_episodes):
            # initial state
            s = env.reset()
            # initial action
            a = action_select(Q, epsilon[i], s)

            # state_check stores the previous state
            # overriddes s in the Q calculation when the game
            # ends
            state_check = s


            done = False
            while not done:
                s_, reward, done = env.step(a)

                # # Improvements to the Q matrix that improves 
                # # rewards by ending Q in the right 
                # # state to separate reward state and lose state
                # # Comment out this if else statement to return the
                # # expected value matrix.
                # if ((done and reward == 0) or (done and reward == 1)):
                #     #final state updated to correct state
                #     s_ = state_check
                # else:
                #     state_check = s_

                a_ = action_select(Q, epsilon[i], s_)


                Q[s, a] += alpha[i] * (reward + (gamma * Q[s_, a_]) - Q[s, a])
                # update processing bar
                if done:
                        break
                
                s, a = s_, a_

            env.reset()

    policy = Q.argmax(axis=1)
    value = Q.max(axis=1)

    return policy, value

   
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    Q = np.zeros((env.n_states, env.n_actions))
    s = env.reset()

    # state_check stores the previous state
    # overriddes s in the Q calculation when the game
    # ends
    state_check = s

    
    for i in range(max_episodes):
        s = env.reset()
        done = False
        
        while not done:
            a = action_select(Q, epsilon[i], s)
            s_, reward, done = env.step(a)
            
            # # Improvements to the Q that improves the 
            # # values and performance by ending Q in the right 
            # # state to separate reward state and lose state
            # # Comment out this if else statement to return the
            # # expected value matrix.
            # if ((done and reward == 0) or (done and reward == 1)):
            #     #final state updated to correct state
            #     s_ = state_check
            # else:
            #     state_check = s_
            
            Q[s][a] = Q[s][a] + eta[i] * ((reward+gamma*Q[s_][np.random.choice(np.flatnonzero(Q[s_, :] == Q[s_, :].max()))]) - Q[s][a])
            s = s_
            
    
    policy = Q.argmax(axis=1)
    value = Q.max(axis=1)
   
    return policy, value

################ Non-tabular model-free algorithms ################

def randomAction(random_state, average_r):
    actions = np.array(np.argwhere(average_r == np.amax(average_r))).flatten()
    return random_state.choice(actions, 1)[0] 

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

