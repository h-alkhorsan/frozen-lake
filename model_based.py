import numpy as np

################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    n = 0

    while n < max_iterations:
        delta = 0
        for state in env.n_states:
            value[state]

        n += 1
        if delta < theta:
            break 

    # TODO:

    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    # TODO:

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    # TODO:
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    # TODO:

    return policy, value