import random
from collections import namedtuple

Transition = namedtuple('Transition' , ('state' , 'action' , 'state_next' , 'reward') )

ENV = 'CartPole-v0'
GAMMA = 0.99  # 時間割引率
MAX_STEPS = 200
NUM_EPISODE = 500


class ReplayMemory:
    def __init__(self , CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self , state , action , state_next , reward):
        if len(self.memory) < self.capacity :
            self.memory.append ( None )
        self.memory[self.index] = Transition ( state , action , state_next , reward )
        self.index = (self.index + 1) % self.capacity

    def sample(self , batch_size):
        return random.sample ( self.memory , batch_size )

    def __len__(self):
        return  len(self.memory)
