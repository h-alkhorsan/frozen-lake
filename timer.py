from model_based import policy_iteration, value_iteration
from frozen_lake import FrozenLake
from model_free import *


import time

lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '#', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '#', '.', '.'],
        ['.', '.', '.', '#', '.', '.', '.', '.'],
        ['.', '#', '#', '.', '.', '.', '#', '.'],
        ['.', '#', '.', '.', '#', '.', '#', '.'],
        ['.', '.', '.', '#', '.', '.', '.', '$']]

seed = 0

env = FrozenLake(lake, slip=0.1, max_steps=64, seed=seed)

gamma = 0.9
theta = 0.001
max_iterations = 100

start_time = time.time()
policy, value = policy_iteration(env, gamma, theta, max_iterations)
end_time = time.time()

env.render(policy,value)

print(f"Time taken for policy iteration: {end_time - start_time}")


start_time = time.time()
policy, value = value_iteration(env, gamma, theta, max_iterations)
end_time = time.time()
env.render(policy,value)

print(f"Time taken for value iteration: {end_time - start_time}")

# policy,value = sarsa(env, 10000, 0.99, gamma, 1)
# env.render(policy,value)

policy,value = q_learning(env, 20000, 0.99, gamma, 1)
env.render(policy,value)


