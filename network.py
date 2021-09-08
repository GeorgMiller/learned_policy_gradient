import keras
import tensorflow as tf
import numpy as np

from keras import Model
from keras.layers import Dense, Input, Conv2D, Flatten


class Net(Model):

    def __init__(self, config):

        super(Net, self).__init__()

        self.config = config 

        self.flatten = Flatten()
        self.dense_1 = Dense(512, activation="relu")
        self.dense_2 = Dense(512, activation="relu")
        self.actor_head = Dense(4, activation="softmax")
        self.critic_head = Dense(1, activation="linear")

    def call(self, inputs):

        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        actor_output = self.actor_head(x)
        critic_output = self.critic_head(x)

        return actor_output, critic_output