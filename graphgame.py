import numpy as np 
import torch


class GraphGame():

    def __init__(self, size, mode, seed):

        self.size = size
        self.mode = mode
        self.seed = seed
        np.random.seed(42)

    def initialize(self):

        self.reward = 0
        self.steps = 0
        self.done = False

        self.state = np.zeros((3, self.size, self.size))
        self.location = torch.randint(self.size - 1, size = (2,))
        self.goal  = torch.randint(self.size - 1, size = (2,))
        self.mine  = torch.randint(self.size - 1, size = (2,))

        while self.location[0] == self.goal[1] & self.location[1] == self.goal[1]:
            self.goal = torch.randint(self.size, size = (2,))

        while self.goal[0] == self.mine[1] & self.goal[1] == self.mine[1]:
            self.mine = torch.randint(self.size, size = (2,))
        
        p1 = np.zeros((3, self.size, self.size))
        self.state[0, self.location[0], self.location[1]] = 1
        self.state[1, self.goal[0], self.goal[1]]  = 1
        self.state[2, self.mine[0], self.mine[1]]  = 1 

        return self.state, self.reward, self.done


    def step(self, action):

        self.steps += 1

        if action ==0:
            if self.location[0] < self.size - 1:
                self.location[0] += 1

        elif action == 1:

            if self.location[1] < self.size - 1:
                self.location[1] += 1

        elif action == 2:
           
            if self.location[0] > 0:
                self.location[0] -= 1

        elif action == 3:

            if self.location[1] > 0:
                self.location[1] -= 1

        self.state[0] = np.zeros((self.size, self.size))
        self.state[0, self.location[0], self.location[1]] = 1


        if np.argmax(self.state[0]) == np.argmax(self.state[1]):
            self.reward = -1
            self.done = True
            #print("Agent won the game after {} steps".format(self.steps))
            #print(np.argmax(self.state[0]), np.argmax(self.state[1]))

        elif np.argmax(self.state[0]) == np.argmax(self.state[2]):
            self.reward = -1
            self.done = True
            #print("Agent went over the mine after {} steps".format(self.steps))
            #print(np.argmax(self.state[0]), np.argmax(self.state[2]))


        else:
            self.reward = 0
        
        
        if self.steps > 30:
            self.done = True

        
        return self.state, self.reward, self.done

