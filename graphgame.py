import numpy as np 


class GraphGame():

    def __init__(self, size, mode, seed):

        self.size = size
        self.mode = mode
        self.seed = seed


        # Dont change the location of the mine and the goal yet.
        np.random.seed(self.seed)   

    def initialize(self):



        self.reward = 0
        self.steps = 0
        self.done = False

        self.state = np.zeros((3, self.size, self.size))
        self.location = np.random.randint(self.size - 1, size = 2)
        self.goal  = np.random.randint(self.size - 1, size = 2)
        self.mine  = np.random.randint(self.size - 1, size = 2)

        while self.location[0] == self.goal[1] & self.location[1] == self.goal[1]:
            self.goal = np.random.randint(self.size, size = 2)

        while self.goal[0] == self.mine[1] & self.goal[1] == self.mine[1]:
            self.mine = np.random.randint(self.size, size = 2)
        
        self.state[0, self.location[0], self.location[1]] = 1
        self.state[1, self.goal[0], self.goal[1]]  = 1
        self.state[2, self.mine[0], self.mine[1]]  = 1 

        return self.state, self.reward, self.done


    def step(self, action):

        self.steps += 1

        if action == 0:
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
            self.reward = 1
            self.done = True
            #print("Agent won the game after {} steps".format(self.steps))
            #print(np.argmax(self.state[0]), np.argmax(self.state[1]))

        elif np.argmax(self.state[0]) == np.argmax(self.state[2]):
            self.reward = -1
            #print("Agent went over the mine after {} steps".format(self.steps))
            #print(np.argmax(self.state[0]), np.argmax(self.state[2]))


        else:
            self.reward = 0
        
        if self.steps > 20:
            self.done = True

        
        return self.state, self.reward, self.done

