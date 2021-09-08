import keras
import tensorflow as tf
import numpy as np

from replaybuffer import Memory
from network import Net
from graphgame import GraphGame
from tensorflow.keras.optimizers import Adam


class Agent():

    def __init__(self, size, mode, seed, config):

        self.graphgame = GraphGame(size, mode, seed)
        self.network = Net(config)

        self.memory = Memory()
        self.episodes = 1000000
        self.batch_size = 32
        self.gamma = 0.95
        self.lr = 0.0001
        self.zero_fixer = 1e-9
        self.seed = 42

        self.optimizer = Adam(learning_rate=self.lr)

    def train(self):

        for episode in range(self.episodes):

            state, reward, done = self.graphgame.initialize()
            # Flatten the state for now. Later add convolutions
            state = np.reshape(state,(1,-1))

            while not done:
                
                action = self.network(state)[0][0]
                actions = [0,1,2,3]
                action_taken = np.random.choice(actions, p=action.numpy())
                next_state, reward, done = self.graphgame.step(action_taken)

                # Flatten the state for now. Later add convolutions
                next_state = np.reshape(next_state,(1,-1))
                action = tf.one_hot(action_taken, 4)
                transition = [state, action, reward, next_state, done]
                self.memory.store(transition)
                state = next_state

                if done:
                    if episode >= 10: 
                        self.update_networks(episode)


    def update_networks(self, episode):

        with tf.GradientTape() as tape:

            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

            state_batch = np.reshape(state_batch,(self.batch_size,-1))
            next_state_batch = np.reshape(next_state_batch,(self.batch_size,-1))

            pred_action, values = self.network(state_batch)
            pred_next_action, next_value = self.network(next_state_batch)

            next_value = tf.reshape(next_value, -1)
            values = tf.reshape(values, -1)

            entropy_coeff = 0.1
            z0 = tf.reduce_sum(pred_action, axis = 1)
            z0 = tf.stack([z0,z0,z0,z0], axis=-1)
            p0 = pred_action / (z0 + self.zero_fixer) 
            entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
            mean_entropy = tf.reduce_mean(entropy) 
            entropy_loss =  mean_entropy * entropy_coeff 

            advantage = reward_batch - values + self.gamma * (next_value) * np.invert(done_batch).astype(np.float32)
            
            log_pred_action = - tf.math.log(tf.reduce_sum(tf.math.multiply(pred_action+self.zero_fixer, action_batch),axis=1))
            actor_loss = tf.reduce_mean(advantage * log_pred_action) + entropy_loss
            critic_loss = tf.reduce_mean(tf.square(reward_batch - values))

            total_loss = 0.5 * critic_loss + actor_loss

            gradients = tape.gradient(total_loss, self.network.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        if episode % 50 == 0:

            print("Episode: {}, reward: {}, actor_loss: {}, critic_loss: {}".format(episode, tf.reduce_sum(reward_batch), actor_loss, critic_loss))

        if episode % 200 == 0:
            
            np.random.seed(self.seed)   

            print(pred_action, values)



agent = Agent(4, mode = None, seed = 42, config = ["empty"])

agent.train()