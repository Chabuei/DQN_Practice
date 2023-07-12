import dezero.functions as F
import numpy as np
import copy
from dezero import optimizers
from DQNBuffer import DQNBuffer
from DQN import DQN

class DQNAgent:
    def __init__(self):
        self.gamma = 0.95
        self.lr = 0.002
        self.epsilon = 0.15
        self.buffer_size = 10000
        self.batch_size = 40
        self.action_size = 2

        self.replay_buffer = DQNBuffer(self.buffer_size, self.batch_size)
        self.qnet = DQN(self.action_size)
        self.qnet_target = DQN(self.action_size) #q関数では教師の値が変わってしまい、それを防ぐためにクローンとして変わらないものを用意しておく(ただし、適度に同期させないと学習できないので少ない回数で同期させることで、教師が変わりすぎてしまうのを防ぐ)
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state): #ε-greedyによりεの確率でargmax以外を選ぶ
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)

            return qs.data.argmax()
        
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.enqueue(state, action, reward, next_state, done)

        if(len(self.replay_buffer) < self.batch_size): #bufferの容量を超えてしまう場合はそれ以上何もしない
            return 
        
        state, action, reward, next_state, done = self.replay_buffer.dequeue()
        qs = self.qnet(state) #入力したstateにおける各actionの収益の期待値が返される(batch_size込み)
        q = qs[np.arange(self.batch_size), action] #qにはqsに入った各バッチにおけるagentがとったactionの収益の期待値が入るようになる(actionには各バッチにおけるagentが取ったactionが0 or 1で入っている)

        next_qs = self.qnet_target(next_state) #次のstateにおける行動価値関数をまた求めている。qnet_targetを用いて計算することにより、教師となるTDターゲットの変動を防いでいる
        next_q = next_qs.max(axis = 1) #次のstateにおける最も収益の期待値が大きくなるようなactionの行動価値関数のみを取る
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()