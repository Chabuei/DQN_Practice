import gym
from DQNAgent import DQNAgent

class Main:
    def __init__(self):
        super().__init__()

        self.episodes = 300
        self.sync_interval = 20
        self.env = gym.make('CartPole-v0')
        self.agent = DQNAgent()
        self.reward_history = []

        self.calculate()

    def calculate(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if isinstance(state, tuple):
                    state = state[0]

                action = self.agent.get_action(state)
                next_step = self.env.step(action)

                next_state = next_step[0]
                reward = next_step[1]
                done = next_step[2]

                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if(episode % self.sync_interval == 0):
                    self.agent.sync_qnet()

                self.reward_history.append(total_reward)

            self.show_result(episode, total_reward)
                
    def show_result(self, episode, total_reward):
        print('episode-' + str(episode + 1) + ': ' +str(total_reward))

Main()