#import packages
import random
from collections import deque

# Use double ended queue to implement the memory pool: D

class Memory:

    def __init__(self):
        self.D = deque(maxlen=2000)
    
    def memorize(self, state, action, reward, next_state, done):
        # when deque is full and new item is appened to the tail
        # a item is discarded from the head
        self.D.append((state, action, reward, next_state, done)) 
        
    def sample_minibatch(self, batch_size):
        minibatch = random.sample(self.D, batch_size)
        return minibatch

    def poollen(self):
        return len(self.D)
