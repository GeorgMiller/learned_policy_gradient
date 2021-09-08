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
        self.episodes = 1000
        self.batch_size = 32
        self.gamma = 0.95
        self.lr = 0.001

        self.optimizer = Adam(learning_rate=self.lr)

    def train(self):

        for episode in range(self.episodes):

            state, reward, done = self.graphgame.initialize()
            
            while not done:

                action, _ = self.network(state)
                next_state, reward, done = self.graphgame.step(action)
                transition = [state, action, reward, next_state, done]
                self.memory.store(transition)
                state = next_state


                if done: 

                    if episode >= 10:

                        with tf.GradientTape() as tape:

                            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
                            pred_action, value = self.network(state_batch)
                            pred_next_action, next_value = self.network(next_state_batch)

                            done_batch = np.invert(done_batch)
                            done_batch = map(int, done_batch)
                            done_batch = list(done_batch)

                            advantage = reward_batch + self.gamma * (value - next_value) * done_batch
                            entropy = 0.5

                            actor_loss = tf.reduce_mean(advantage * tf.math.log(pred_action)) + entropy
                            critic_loss = tf.reduce_mean(tf.square(reward_batch - value))

                            total_loss = 0.5 * critic_loss + actor_loss

                            gradients = tape.gradient(total_loss, self.network.trainable_weights)
                            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

                        print("Episode: {}, reward: {}, actor_loss: {}, critic_loss: {}".format(episode, reward, actor_loss, critic_loss))




agent = Agent((6,6), mode = None, seed = 42, config = ["empty"])

agent.train()