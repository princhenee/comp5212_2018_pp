import tensorflow as tf

class Model:
    def __init__(self):
        raise NotImplementedError

    def inference(self, X):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def initialize_parameters(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def train_operation(self):
        raise NotImplementedError

    def train(self, X, Y):
        raise NotImplementedError

    def loss_function(self):
        raise NotImplementedError

