import keras
import tensorflow as tf
import numpy as np

from keras import Model
from keras.layers import Dense, Input, Conv2D, Flatten, LSTM


class A2C(Model):

    def __init__(self, config):

        super(A2C, self).__init__()

        self.config = config 

        self.dense_1 = Dense(512, activation="relu")
        self.dense_2 = Dense(512, activation="relu")
        self.actor_head = Dense(4, activation="softmax")
        self.critic_head = Dense(1, activation="relu")

    def call(self, inputs):

        x = self.dense_1(inputs)
        x = self.dense_2(x)
        actor_output = self.actor_head(x)
        critic_output = self.critic_head(x)

        return actor_output, critic_output


class Network(Model):

    def __init__(self, config):

        super(Network, self).__init__()

        self.config = config 

        self.dense_1 = Dense(512, activation="relu")
        self.dense_2 = Dense(512, activation="relu")
        self.policy_head = Dense(4, activation="softmax")
        self.prediction_head = Dense(30, activation="relu")

    def call(self, inputs):

        x = self.dense_1(inputs)
        x = self.dense_2(x)
        policy_output = self.policy_head(x)
        prediction_output = self.prediction_head(x)

        return policy_output, prediction_output


class Meta_Network(Model):

    def __init__(self, config):

        super(Meta_Network, self).__init__()

        self.config = config 

        #self.lstm_1 = LSTM(512, activation="relu")
        self.dense_1 = Dense(512, activation="relu")
        self.policy_head = Dense(4, activation="softmax")
        self.prediction_head = Dense(30, activation="linear")

    def call(self, inputs):

        #x = self.lstm_1(inputs)
        x = self.dense_1(inputs)
        policy_output = self.policy_head(x)
        prediction_output = self.prediction_head(x)

        return policy_output, prediction_output