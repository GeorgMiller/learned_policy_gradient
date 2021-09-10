# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import higher
#from ml3.ml3_train import meta_train_mountain_car as meta_train
#from ml3.ml3_test import test_ml3_loss_mountain_car as test_ml3_loss
#from ml3.learnable_losses import Ml3_loss_mountain_car as Ml3_loss
#from ml3.optimizee import MC_Policy
from mountain_car import MountainCar

def test_ml3_loss_mountain_car(policy, ml3_loss, opt_iter, *args):

    opt = torch.optim.SGD(policy.parameters(), lr=policy.learning_rate)
    for i in range(opt_iter):
        s_tr, a_tr, g_tr = policy.roll_out(*args)
        pred_task_loss = ml3_loss(torch.cat([s_tr[:-1], a_tr, g_tr], dim=1)).mean()
        opt.zero_grad()
        pred_task_loss.backward()
        opt.step()
        s_tr, a_tr, g_tr = policy.roll_out(*args)
        print('last state: ', s_tr[-1])
    return s_tr.detach().numpy()

class MC_Policy(nn.Module):

    def __init__(self, pi_in, pi_out):
        super(MC_Policy, self).__init__()

        num_neurons = 200
        self.policy = nn.Sequential(nn.Linear(pi_in, num_neurons,bias=False),
                                    nn.Linear(num_neurons, pi_out,bias=False))
        self.learning_rate = 1e-3
        self.env = MountainCar()

    def forward(self, x):
        return self.policy(x)

    def reset_gradients(self):
        for i, param in enumerate(self.policy.parameters()):
            param.detach()

    def roll_out(self, s_0, goal, time_horizon):
        state = torch.Tensor(self.env.reset_to(s_0))
        states = []
        actions = []
        states.append(state)
        for t in range(time_horizon):

            u = self.forward(state)
            u = u.clamp(self.env.min_action, self.env.max_action)
            state = self.env.sim_step_torch(state.squeeze(), u.squeeze()).clone()
            states.append(state.clone())
            actions.append(u.clone())

        running_reward = torch.norm(state-goal)
        rewards = [torch.Tensor([running_reward])]*time_horizon
        return torch.stack(states), torch.stack(actions), torch.stack(rewards)

class Ml3_loss_mountain_car(nn.Module):

    def __init__(self, meta_in, meta_out):
        super(Ml3_loss_mountain_car, self).__init__()

        activation = torch.nn.ELU
        num_neurons = 400
        self.loss_fn = torch.nn.Sequential(torch.nn.Linear(meta_in, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, meta_out))
        self.learning_rate = 1e-3

    def forward(self, x):
        return self.loss_fn(x)

def meta_train_mountain_car(policy,ml3_loss,task_loss_fn,s_0,goal,goal_extra,n_outer_iter,n_inner_iter,time_horizon,shaped_loss):
    s_0 = torch.Tensor(s_0)
    goal = torch.Tensor(goal)
    goal_extra = torch.Tensor(goal_extra)

    inner_opt = torch.optim.SGD(policy.parameters(), lr=policy.learning_rate)
    meta_opt = torch.optim.Adam(ml3_loss.parameters(), lr=ml3_loss.learning_rate)

    for outer_i in range(n_outer_iter):
        # set gradient with respect to meta loss parameters to 0
        meta_opt.zero_grad()
        for _ in range(n_inner_iter):
            inner_opt.zero_grad()
            with higher.innerloop_ctx(policy, inner_opt, copy_initial_weights=False) as (fpolicy, diffopt):
                # use current meta loss to update model
                s_tr, a_tr, g_tr = fpolicy.roll_out(s_0, goal, time_horizon)

                loss_input = torch.cat([s_tr[:-1], a_tr, g_tr], dim=1)
                pred_task_loss = ml3_loss(loss_input).mean()
                diffopt.step(pred_task_loss)

            # compute task loss
            s, a, g = fpolicy.roll_out(s_0, goal, time_horizon)
            task_loss = task_loss_fn(a, s[:], goal, goal_extra, shaped_loss)
            # backprop grad wrt to task loss
            task_loss.backward()

        meta_opt.step()

        if outer_i % 100 == 0:
            print("meta iter: {} loss: {}".format(outer_i, task_loss.item()))
            print('last state', s[-1])



EXP_FOLDER = os.path.join("experiments/data/mountain_car")


class Task_loss(object):
    def __call__(self, a, s, goal, goal_exp,shaped_loss):

        loss = (torch.norm(s - goal)).mean()
        if shaped_loss:
            loss = (torch.norm(s[:15] - goal_exp)).mean() + (torch.norm(s[15:] - goal)).mean()

        return loss


if __name__ == '__main__':

    if not os.path.exists(EXP_FOLDER):
        os.makedirs(EXP_FOLDER)

    np.random.seed(0)
    torch.manual_seed(0)

    policy = MC_Policy(2,1)
    ml3_loss = Ml3_loss_mountain_car(4,1)

    task_loss = Task_loss()

    goal = [0.5000, 1.0375]
    goal_extra = [-0.9470, -0.0055]

    env = MountainCar()
    s_0 = env.reset()

    n_outer_iter = 300
    n_inner_iter = 1

    time_horizon = 35

    #if sys.argv[1] == 'train':
    shaped_loss = 'True' # sys.argv[2] == 'True'
    meta_train_mountain_car(policy, ml3_loss, task_loss, s_0, goal, goal_extra, n_outer_iter, n_inner_iter, time_horizon, shaped_loss)
    if shaped_loss:
        torch.save(ml3_loss.state_dict(), f"{EXP_FOLDER}/shaped_ml3_loss_mountain_car.pt")
    else:
        torch.save(ml3_loss.state_dict(), f"{EXP_FOLDER}/ml3_loss_mountain_car.pt")
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