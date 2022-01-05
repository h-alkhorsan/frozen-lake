from frozen_lake import *

lake =  [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

env = FrozenLake(lake, slip=0.1, max_steps=16)

probability_matrix = np.load('p.npy')

is_equal = np.array_equal(env.p_matrix, probability_matrix)

# debug
if not is_equal:
    debug_array = np.subtract(env.p_matrix, probability_matrix)
    incorrect_vals = np.where(debug_array > 0)
    print(debug_array[incorrect_vals])
else:
    print("Correct implementation")



   




