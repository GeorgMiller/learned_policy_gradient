import numpy as np 
import random


class Memory():

    def __init__(self):

        self.size = 1000
        self.memory = []
    
    def store(self, transition):

        if len(self.memory) >= self.size:
            self.memory.pop(0)
        
        self.memory.append(transition)

    def sample(self, batch_size):

        samples = random.sample(self.memory, batch_size)

        return map(np.array, zip(*samples))