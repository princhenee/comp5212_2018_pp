import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque


class DeepQAgent:
    def __init__(self, model_name, save_path,
                        state_shape, action_size, action):
        self._model_name = model_name
        self._save_path = save_path
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0    # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Car Agent
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(
                                      input_shape=self.state_shape,
                                      filters=32,
                                      kernel_size=(3,3),
                                      strides=1,
                                      padding='same',
                                      activation = 'relu'
                                     )
                  )
        model.add(keras.layers.MaxPool2D(
                                         pool_size=(2,2),
                                         strides=2,
                                         padding='same',
                                        )
                  )
        model.add(keras.layers.Conv2D(
                                      filters=32,
                                      kernel_size=(3,3),
                                      strides=2,
                                      padding='same',
                                      activation = 'relu'
                                     )
                  )
        model.add(keras.layers.Conv2D(
                                      filters=64,
                                      kernel_size=(3,3),
                                      strides=1,
                                      padding='same',
                                      activation = 'relu'
                                     )
                  )
        model.add(keras.layers.MaxPool2D(
                                         pool_size=(2,2),
                                         strides=2,
                                         padding='same',
                                        )
                  )
        model.add(keras.layers.dropout(0.25))    # reduce overfitting

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))

        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model


    def inference(self, state):
        pass

    def parameters(self):
        # return list(self._parameters.values())
        pass

    def initialize_parameters(self):
        '''
        with tf.Session() as sess:
            sess.run(tf.initialize_variables(self.parameters()))
        '''
        pass

    def save(self):
        '''
        saver = tf.train.Saver(
            self.parameters(),
            save_relative_paths=True,
            filename=self._model_name)
        saver.save(sess, self._save_path)
        '''
        self.model.save_weights(self._save_path)

    def load(self):
        '''
        saver = tf.train.Saver(
            self.parameters(),
            save_relative_paths=True,
            filename=self._model_name)
        saver.restore(sess, self._save_path)
        '''
        self.model.load_weights(self._save_path)

    def train_operation(self):
        pass

    def train(self, X, Y):
        pass

    def loss_function(self):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.inference(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                        np.amax(self.model.predict(next_state)[0])
            target_f = self.inference(state)
            target_f[0][action] = target
            self.train(state, target_f, epochs=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

