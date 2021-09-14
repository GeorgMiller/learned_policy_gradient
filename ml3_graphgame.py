# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import higher
from graphgame import GraphGame


class Policy_Network(nn.Module):

    def __init__(self, pi_in, pi_out):
        super(Policy_Network, self).__init__()

        num_neurons = 256
        self.policy = nn.Sequential(nn.Linear(pi_in, num_neurons,bias=False),
                                    nn.Linear(num_neurons, pi_out,bias=False),
                                    nn.Softmax(dim=None))
        self.learning_rate = 1e-3


        self.graphgame = GraphGame(6, None, 42)

    def forward(self, x):

        return self.policy(x)

    def reset_gradients(self):

        for i, param in enumerate(self.policy.parameters()):
            param.detach()

    def play_episode_inner(self, time_horizon):

        state, reward, done = self.graphgame.initialize()
        state = torch.Tensor(state)

        state = torch.flatten(state)
        print(state, reward)

        states = []
        actions = []
        rewards = []

        state = torch.Tensor(state)

        for t in range(time_horizon):

            prediction = self.forward(state)
            
            #action_taken = torch.argmax(action)
            action_taken = torch.multinomial(prediction,1)


            next_state, reward, done = self.graphgame.step(action_taken)


            next_state = torch.Tensor(next_state)
            reward = torch.Tensor([reward])            
            next_state = torch.flatten(next_state)

            
            action_taken = torch.nn.functional.one_hot(action_taken, 4)[0]
            action = action_taken

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        return torch.stack(states), torch.stack(actions), torch.stack(rewards)

    def play_episode_outer(self, time_horizon):

        state, reward, done = self.graphgame.initialize()
        state = torch.Tensor(state)

        state = torch.flatten(state)
        print(state, reward)

        states = []
        actions = []
        rewards = []

        state = torch.Tensor(state)

        for t in range(time_horizon):

            prediction = self.forward(state)
            
            #action_taken = torch.argmax(action)
            action_taken = torch.multinomial(prediction,1)


            next_state, reward, done = self.graphgame.step(action_taken)


            next_state = torch.Tensor(next_state)
            reward = torch.Tensor([reward])            
            next_state = torch.flatten(next_state)

            
            action_taken = torch.nn.functional.one_hot(action_taken, 4)[0]
            action = prediction * action_taken

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        return torch.stack(states), torch.stack(actions), torch.stack(rewards)

class Meta_Network(nn.Module):

    def __init__(self, meta_in, meta_out):
        super(Meta_Network, self).__init__()

        activation = torch.nn.ELU
        num_neurons = 512
        self.loss_fn = torch.nn.Sequential(torch.nn.Linear(meta_in, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, meta_out))
        self.learning_rate = 1e-3

    def forward(self, x):
        return self.loss_fn(x)




def train(policy_network, meta_network, n_outer_iter, n_inner_iter, time_horizon):


    task_loss_fn = Task_loss()
    policy_opt = torch.optim.SGD(policy_network.parameters(), lr=policy_network.learning_rate)
    meta_opt = torch.optim.Adam(meta_network.parameters(), lr=meta_network.learning_rate)

    for outer_i in range(n_outer_iter):

        # set gradient with respect to meta loss parameters to 0
        meta_opt.zero_grad()
        for _ in range(n_inner_iter):

            policy_opt.zero_grad()

            with higher.innerloop_ctx(policy_network, policy_opt, copy_initial_weights=False) as (f_policy, diff_opt):

                # use current meta loss to update model
                states, actions, rewards = f_policy.play_episode_inner(time_horizon)

                loss_input = torch.cat([states, actions, rewards], dim=1)
                pred_task_loss = meta_network(loss_input).mean()
                diff_opt.step(pred_task_loss)

            # compute task loss
            state, act, reward = f_policy.play_episode_outer(time_horizon)

            location = torch.argmax(state[0]).float()
            goal = torch.argmax(state[1]).float()
            
            max_reward = torch.Tensor([1.])
            max_rewards = [torch.Tensor([max_reward])]*time_horizon
            max_rewards = torch.stack(max_rewards)
            task_loss = - (torch.log(act)).mean()
            # backprop grad wrt to task loss
            task_loss.backward()

        meta_opt.step()

        if outer_i % 100 == 0:
            print("meta iter: {} loss: {}".format(outer_i, task_loss.item()))
            #print('last state', s[-1])


class Task_loss(object):
    def __call__(self, reward, max_reward):

        loss = (torch.norm(reward - max_reward)).mean()
        return loss



if __name__ == '__main__':

    EXP_FOLDER = os.path.join("experiments/data/graphgame")

    if not os.path.exists(EXP_FOLDER):
        os.makedirs(EXP_FOLDER)

    np.random.seed(0)
    torch.manual_seed(0)

    meta_in = 113     # (state, action, reward) - tuple. state: dim=2
    meta_out = 1    # single output node for loss (should be more for more complex problems?)

    policy_in = 108   # state size
    policy_out = 4  # action size

    policy_network = Policy_Network(policy_in, policy_out)
    meta_network = Meta_Network(meta_in, meta_out)

    n_outer_iter = 300
    n_inner_iter = 1

    time_horizon = 10   # Number of time steps to unroll 

    train(policy_network, meta_network, n_outer_iter, n_inner_iter, time_horizon)
    torch.save(meta_network.state_dict(), f"{EXP_FOLDER}/ml3_loss_mountain_car.pt")
    
    
    
    '''
    #if sys.argv[1] == 'test':
    shaped_loss = 'True' # sys.argv[2] == 'True'
    if shaped_loss:
        ml3_loss.load_state_dict(torch.load(f"{EXP_FOLDER}/shaped_ml3_loss_mountain_car.pt"))
    else:
        ml3_loss.load_state_dict(torch.load(f"{EXP_FOLDER}/ml3_loss_mountain_car.pt"))
    ml3_loss.eval()
    opt_iter = 2
    args = (torch.Tensor(s_0), torch.Tensor(goal), time_horizon)
    states = test_ml3_loss_mountain_car(policy, ml3_loss, opt_iter, *args)
    if shaped_loss:
        np.save(f"{EXP_FOLDER}/shaped_ml3_mc_states.npy", states)
    else:
        np.save(f"{EXP_FOLDER}/ml3_mc_states.npy", states)

    if shaped_loss:
        env.render(list(np.array(states)[:, 0]), file_path=f"{EXP_FOLDER}/shaped_ml3_mc.gif")
    else:
        env.render(list(np.array(states)[:, 0]), file_path=f"{EXP_FOLDER}/ml3_mc.gif")
    '''