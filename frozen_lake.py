from model import *

"""
lake: A matrix that represents the lake. For example:
    lake =  [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
slip: The probability that the agent will slip
max_steps: The maximum number of time steps in an episode
seed: A seed to control the random number generator (optional)
"""

class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        # TODO:
        super(FrozenLake, self).__init__(n_states, n_actions, max_steps, pi, seed)

        self.p_matrix = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)
        self.compute_p()
        self.rewards = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)
        self.compute_r()

    def compute_p(self):
        terminal_state = ['#', '$']
        for state in range(self.n_states):
            if state == self.absorbing_state or self.lake_flat[state] in terminal_state:
                self.p_matrix[self.absorbing_state, state, :] = 1
                continue
            for i in range(self.n_actions):
                for j in range(self.n_actions):
                    next_state = self.action(state, j)
                    self.p_matrix[next_state, state, i] += self.slip / self.n_actions
                    if i == j:
                        self.p_matrix[next_state, state, i] += 1 - self.slip


    def compute_r(self):
        for state in range(self.n_states):
            if state != self.absorbing_state:
                if self.lake_flat[state] == '$':
                #and self.lake_flat[state] == '$':
                    self.rewards[self.absorbing_state, state, :] = 1


    def action(self, state, action):
        # move up
        if action == 0:
            if state - self.lake.shape[1] not in range(self.n_states):
                return state 
            next_state = state - self.lake.shape[1]

        # move left
        if action == 1:
            if state - 1 not in range(self.n_states):
                return state 
            next_state = state - 1

        # move down
        if action == 2:
            if state + self.lake.shape[1] not in range(self.n_states):
                return state
            next_state = state + self.lake.shape[1]

        # move right
        if action == 3:
            if state + 1 not in range(self.n_states):
                return state
            next_state = state + 1

        return next_state



        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        # TODO:
        return self.p_matrix[next_state, state, action]

    def r(self, next_state, state, action):
        # TODO:
        return self.rewards[next_state, state, action]

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))
