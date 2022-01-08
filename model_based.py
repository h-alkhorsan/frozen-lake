import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float) 
    n = 0 
    while n < max_iterations: 
        delta = 0
        for state in range(env.n_states): 
            v = value[state] 

            value[state] = sum([
                            env.p(ns,state,policy[state]) * 
                            (env.r(ns,state,policy[state]) + 
                            gamma * 
                            value[ns]) for ns in range(env.n_states)])

            delta = max(delta, abs(v - value[state])) 

        if delta < theta: 
            break
        n += 1
    return value


def policy_improvement(env, policy, value, gamma):
    if policy is None: 
        policy = np.zeros(env.n_states, dtype=int) 
    else:
        policy = np.array(policy, dtype=int) 

    stable = True
     
    for state in range(env.n_states): 
        current = policy[state] 

        get_actions = env.a(state) 
        actions = [get_actions.index(a) for a in get_actions]

        policy[state] = actions[int(np.argmax([
                                sum([
                                env.p(ns, state, a) * 
                                (env.r(ns, state, a) + 
                                gamma * 
                                value[ns]) for ns in range(env.n_states)]) for a in actions]))]
                                
        if current != policy[state]:
            stable = False 

    return policy, stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    n = 0

    stable = False
    while not stable: 
        values = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy, stable = policy_improvement(env, policy, values, gamma)
        n += 1

    print(f"Iterations: {n}")
    return policy, values


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states) 
    else:
        value = np.array(value, dtype = np.float) 

    n = 0 

    while n < max_iterations:  
        delta = 0 
        for state in range(env.n_states): 
            get_actions = env.a(state) 
            actions = [get_actions.index(a) for a in get_actions]

            v = value[state]

            value[state] = max([sum([
                                env.p(ns, state, a) * 
                                (env.r(ns, state, a) + 
                                gamma * 
                                value[ns]) for ns in range(env.n_states)]) for a in actions])

            delta = max(delta, abs(v - value[state])) 

        if delta < theta: 
            break
        n += 1 

    policy = np.zeros((env.n_states), dtype=int) 
    for state in range(env.n_states): 
        get_actions = env.a(state)
        
        actions = [get_actions.index(a) for a in get_actions]
        policy[state] = actions[int(np.argmax([
                                sum([env.p(ns, state, a) * 
                                (env.r(ns, state, a) + 
                                gamma * 
                                value[ns]) for ns in range(env.n_states)]) for a in actions]))]

    print(f"Iterations: {n}")
    return policy, value
