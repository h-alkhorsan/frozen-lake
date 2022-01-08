from model_based import policy_iteration, value_iteration
from frozen_lake import FrozenLake
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

env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

gamma = 0.9
theta = 0.001
max_iterations = 100

start_time = time.time()
policy, value = policy_iteration(env, gamma, theta, max_iterations)
end_time = time.time()

print(f"Time taken for policy iteration: {end_time - start_time}")


start_time = time.time()
policy, value = value_iteration(env, gamma, theta, max_iterations)
end_time = time.time()

print(f"Time taken for value iteration: {end_time - start_time}")