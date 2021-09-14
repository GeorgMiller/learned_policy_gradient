import keras
from keras.backend import log
from keras.mixed_precision import policy
from numpy.lib.function_base import gradient
import tensorflow as tf
import numpy as np



import keras
import tensorflow as tf
import numpy as np

from replaybuffer import Memory
from network import Network, Meta_Network
from graphgame import GraphGame
from tensorflow.keras.optimizers import Adam, SGD

       


class Algorithm():

    def __init__(self, size, mode, seed, config):

        self.meta = Meta_Network(config)

        '''
        Parameters of actual paper:
            - 960 parallel lifetimes (batch size of meta-gradients)
            - each lifetime has a agent which interact with the enviroment for 10^10 steps
            - agent uses a batch of trajectories generated of 64 enviroments (batch size agent)
            - each trajectory consists of 20 steps (batch size = 64 x 20) update every agent after 20 steps
            - y_t and y_t_1 are mapped to a scalar by an embedding network Dense(16)-Dense(1)
            - All layers have ReLU
            - LSTM with 256 units
            - reset the hidden state for terminal states (e.g. d = 0)
            - 'outer' algorithm can be any RL algorithm as long as it maximizes commulative rewards
            - use a bandit for sampling hyperparameters for different enviroments
            - reset the lifetime whenever the entropy of the policy becomes 0

        '''




        # Meta-hyperparameters for meta-training
        self.lr_meta = 0.0001
        self.optimizer_meta = Adam(learning_rate = self.lr_meta)
        self.discount_reward_factor = 0.99
        self.beta_0 = 0.01
        self.beta_1 = 0.001
        self.beta_2 = 0.001 
        self.beta_3 = 0.001

        self.num_parameter_updates = 5 # == K
        self.num_trajectories = 20
        self.batch_size_meta = 64 # original = 960
        self.batch_size_agent = 16 # original = 64

        # Hyperparameters for training the agent
        self.lr_agent = 0.001
        self.optimizer_agent = Adam(learning_rate = self.lr_agent)
        self.life_time_agent = 300_000
        self.kl_cost = 0.1

        self.episodes = 1000000
        self.batch_size = 32
        self.gamma = 0.95
        self.zero_fixer = 1e-9
        self.seed = 42


    def train(self):

        self.agents = {}

        for idx in range(self.batch_size_meta):

            self.agents[idx] = [Network(), GraphGame(), 0 for _ in range(self.batch_size_agent)]




    def play_episode(self):

        
        for agent in self.agents:

            if agent.life_time_agent <= 300_000:

                agent.play_episode()


        for episode in range(self.episodes):

            state, reward, done = self.graphgame.initialize()
            # Flatten the state for now. Later add convolutions
            state = np.reshape(state,(1,-1))

            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                
                action = self.agent(state)[0][0]
                actions_game = [0,1,2,3]
                action_taken = np.random.choice(actions_game, p=action.numpy())
                next_state, reward, done = self.graphgame.step(action_taken)

                # Flatten the state for now. Later add convolutions
                next_state = np.reshape(next_state,(1,-1))
                action = tf.one_hot(action_taken, 4).numpy()

                # Reward needs to be reshaped, otherwise it probably doesn't work too good.

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state



                if done:

                    discounted_rewards = []
                    last_reward = rewards[-1]
                    step = 0
                    for reward in reversed(rewards):
                        step += 1
                        if step == 1:
                            discounted_rewards.append(last_reward)
                        else:
                            last_reward = last_reward * self.discount_reward_factor
                            discounted_rewards.insert(0, last_reward)
                    
                    discounted_rewards = np.array(discounted_rewards)

                    #if np.sum(discounted_rewards) != 0:
                    #    discounted_rewards -= np.mean(discounted_rewards)
                    #    discounted_rewards /= np.std(discounted_rewards)

                    for state, action_2, reward_2, next_state, done_2 in zip(states, actions, discounted_rewards, next_states, dones):
                        transition = [state, action_2, reward_2, next_state, done_2]
                        self.memory.store(transition)

                    if episode >= 10:

                        for _ in range(10): 
                            self.update_agent(episode)

                        self.update_meta()


    def update_agent(self, episode):

        with tf.GradientTape() as tape:

            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

            state_batch = np.reshape(state_batch,(self.batch_size,-1))
            next_state_batch = np.reshape(next_state_batch,(self.batch_size,-1))


            policy_estimate, prediction_estimate = self.agent(state_batch)
            _, prediction_estimate_next_state = self.agent(next_state_batch)

            policy_action_batch = tf.reduce_sum(tf.math.multiply(policy_estimate+self.zero_fixer, action_batch),axis=1)

            meta_input = []
            inputs = []

            for i in range(self.batch_size):
                inputs.append(reward_batch[i])
                inputs.append(done_batch[i].astype(np.float32))
                inputs.append(policy_action_batch[i])
                for s_0 in prediction_estimate[i]:

                    inputs.append(s_0)
                for s_1 in prediction_estimate_next_state[i]:
                    inputs.append(s_1)

                #inputs = tf.squeeze(inputs)
            meta_input = tf.reshape(inputs, (self.batch_size, -1))
            
            policy_target, prediction_target = self.meta(meta_input)

            policy_target = tf.reduce_sum(tf.math.multiply(policy_target+self.zero_fixer, action_batch),axis=1)

            prediction_estimate = tf.reshape(prediction_estimate, (self.batch_size, -1))
            prediction_estimate_next_state = tf.reshape(prediction_estimate_next_state, (self.batch_size, -1))

            log_policy_estimate = tf.math.log(policy_action_batch)
            agent_loss = log_policy_estimate * policy_target - tf.reduce_sum(prediction_estimate * tf.math.log(tf.math.abs((prediction_estimate + self.zero_fixer) / prediction_target)), axis = 1)
            agent_loss = tf.reduce_mean(agent_loss)
            gradients = tape.gradient(agent_loss, self.agent.trainable_weights)
            self.optimizer_agent.apply_gradients(zip(gradients, self.agent.trainable_weights))

            print("Episode: {}, reward: {}, actor_loss: {}".format(episode, tf.reduce_sum(reward_batch), agent_loss))

        if episode % 200 == 0:
            
            #np.random.seed(self.seed)   
            x = 0

    def update_meta(self):

        with tf.GradientTape() as tape:

            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

            state_batch = np.reshape(state_batch,(self.batch_size,-1))
            next_state_batch = np.reshape(next_state_batch,(self.batch_size,-1))

            policy_estimate, prediction_estimate = self.agent(state_batch)
            _, prediction_estimate_next_state = self.agent(next_state_batch)

            policy_action_batch = tf.reduce_sum(tf.math.multiply(policy_estimate+self.zero_fixer, action_batch),axis=1)

            meta_input = []
            inputs = []

            for i in range(self.batch_size):
                inputs.append(reward_batch[i])
                inputs.append(done_batch[i].astype(np.float32))
                inputs.append(policy_action_batch[i])
                for s_0 in prediction_estimate[i]:
                    inputs.append(s_0)
                for s_1 in prediction_estimate_next_state[i]:
                    inputs.append(s_1)

            meta_input = tf.reshape(inputs, (self.batch_size, -1))            
            policy_target, prediction_target = self.meta(meta_input)

            z0 = tf.reduce_sum(policy_estimate, axis = 1)
            z0 = tf.stack([z0, z0, z0, z0], axis = 1)
            p0 = policy_estimate / (z0 + self.zero_fixer) 
            entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
            mean_entropy = tf.reduce_mean(entropy) 
            entropy_policy =  mean_entropy * self.beta_0 

            z0 = tf.reduce_sum(prediction_estimate, axis = 1)
            z0 = tf.stack([z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0], axis = 1)            
            p0 = prediction_estimate / (z0 + self.zero_fixer) 
            entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
            mean_entropy = tf.reduce_mean(entropy) 
            entropy_prediction =  mean_entropy * self.beta_0 

            log_policy_estimate = tf.math.log(policy_action_batch)

            G = reward_batch

            meta_loss = tf.reduce_mean(log_policy_estimate * G + self.beta_0 * entropy_policy + self.beta_1 * entropy_prediction \
                        - self.beta_2 * tf.norm(policy_target) - self.beta_3 * tf.norm(prediction_target))

            gradients = tape.gradient(meta_loss, self.meta.trainable_variables)
            self.optimizer_meta.apply_gradients(zip(gradients, self.meta.trainable_variables))

            print(meta_loss, entropy_prediction, entropy_policy)


agent = Algorithm(5, mode = None, seed = 42, config = ["empty"])

agent.train()