import gym
import numpy as np
import torch
import agent

ENV = 'CartPole-v0'
NUM_EPISODES = 500
MAX_STEPS = 200


class Environment:
    def __init__(self):
        self.env =gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.agent = agent.Agent(self.num_states, self.num_actions)

    def run(self):
        episode_10_list = np.zeros(10)
        complete_episode =0
        episode_final =False
        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            state = observation
            state= torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)
                observation_next, _, done, _ = self.env.step(action.item())
                if done:
                    state_next = None
                    episode_10_list = np.hstack((episode_10_list[1:], step+1))
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episode = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episode = complete_episode +1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)
                self.agent.upper_q_function()
                state = state_next

                if done:
                    print('%d Episode Finished after %d steps:10試行の平均steps数=%.1lf'%(episode, step +1, episode_10_list.mean()))
                    break

            if episode_final is True:
                break

            if complete_episode >= 10:
                print('10回連続成功')
                episode_final = True

cartpole_env = Environment()
cartpole_env.run()