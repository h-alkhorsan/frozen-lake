from frozen_lake import *

lake =  [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

env = FrozenLake(lake, slip=0.1, max_steps=16)

play(env)



