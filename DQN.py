import dezero.functions as F
import dezero.layers as L
from dezero import Model

class DQN(Model): #stateを入れるとそのstate(サイズ4)における各action(サイズ2)により得られる収益の期待値が得られる(行動価値関数)
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(150)
        self.l2 = L.Linear(150)
        self.l3 = L.Linear(150)
        self.l4 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)

        return x