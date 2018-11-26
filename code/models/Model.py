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

    def save(self, sess: tf.Session()):
        raise NotImplementedError

    def load(self, sess: tf.Session()):
        raise NotImplementedError

    def sync(self, target: Model):
        raise NotImplementedError

    def copy(self, model_name: str, save_path: str):
        raise NotImplementedError
