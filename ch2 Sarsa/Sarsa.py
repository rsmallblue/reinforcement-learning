import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(3)
N_STATES = 5
ACTIONS = ['left', 'right']
MAX_EPISODES = 200
EPSILON = 0.9
GAMMA = 0.5
ALPHA = 0.1


def reset_env():
    env = ['-']*(N_STATES-1)+['T']
    env[0] = '*'
    print(env)

class qln():
    def __init__(self, actions, n_states):
        self.n_actions = actions
        self.n_states = n_states
        self.q_table = pd.DataFrame(np.zeros((n_states, len(actions))), np.arange(n_states), actions)
        self.step_counter = 0

    def reset_env(self):
        env = ['-']*(N_STATES-1)+['T']
        env[0] = '*'
        print(env)


    def choose_action(self,state):
        state_action = self.q_table.loc[state,:]
        if np.random.uniform() > EPSILON or (state_action == 0).all():
            action_name = np.random.choice(ACTIONS)
        else:
            action_name = state_action.idxmax()
        return action_name

    def update_env(self, state, action_name):
        env = ['-']*(N_STATES-1)+['T']
        if action_name == 'right':
            next_state = state + 1
            if next_state == N_STATES-1:
                reward = 1
            else:
                reward = -0.5
        else:
            if state == 0 :
                next_state = 0
            else:
                next_state = state -1
            reward = -0.5
        env[state] = '*'
        env = ''.join(env)
        print(env)
        return next_state, reward




if __name__ =='__main__':
    ql = qln(ACTIONS, N_STATES)
    for episode in range(MAX_EPISODES):
        reset_env()
        reward = 0
        state = 0
        is_terminal = False
        step = 0
        step_counter = []
        while(not is_terminal):
            action = ql.choose_action(state)
            next_state, reward = ql.update_env(state, action)
            if next_state == N_STATES-1:
                is_terminal = True
                q_target = reward
            else:
                next_action = ql.choose_action(next_state)
                delta = reward + GAMMA * ql.q_table.loc[next_state,next_action] - ql.q_table.loc[state, action]
                ql.q_table.loc[state, action] += ALPHA * delta
            state = next_state
            step += 1
            if is_terminal:
                step_counter.append(step)
    print(ql.q_table)








