import tensorflow as tf
from Model import Model


class CarAgentModel(Model):
    def __init__(self):
        with tf.variable_scope("CarAgent"):
            self._parameters = {
                "conv1_w": tf.get_variable("conv1_w", shape=[3, 3, 3, 32]),
                "conv2_w": tf.get_variable("conv2_w", shape=[2, 2, 32, 32]),
                "conv3_w": tf.get_variable("conv3_w", shape=[2, 2, 32, 64]),
                "conv4_w": tf.get_variable("conv4_w", shape=[2, 2, 64, 128]),
                "relu5_w": tf.get_variable("relu5_w", shape=[19*9*128 + 2, 1024]),
                "relu5_b": tf.get_variable("relu5_b", shape=[1024]),
                "relu6_w": tf.get_variable("relu6_w", shape=[1024, 512]),
                "relu6_b": tf.get_variable("relu6_b", shape=[512])
                "relu7_w": tf.get_variable("relu6_w", shape=[512, 256]),
                "relu7_b": tf.get_variable("relu6_b", shape=[256])
                "relu8_w": tf.get_variable("relu6_w", shape=[256, 256]),
                "relu8_b": tf.get_variable("relu6_b", shape=[256])
                "logit_w": tf.get_variable("logit_w", shape=[256, 256]),
                "logit_b": tf.get_variable("logit_b", shape=[256])
            }
        raise NotImplementedError

    def inference(self, X):
        image_array = tf.convert_to_tensor(X[0])  # (?,320,160,3)
        speed = tf.convert_to_tensor([X[1]])  # [0,1]
        steering_angle = tf.convert_to_tensor([X[2]])  # [-1,1]

        conv1 = tf.nn.conv2d(
            image_array,
            self._parameters["conv1_w"],
            [1, 1, 1, 1],
            "SAME",
            name="conv1")  # (?,318,158,32)
        relu1 = tf.nn.leaky_relu(conv1, name="relu1")
        pool1 = tf.nn.max_pool(
            relu1,
            (1, 2, 2, 1),
            (1, 2, 2, 1),
            "SAME",
            name="pool1")  # (?,159,79,32)
        conv2 = tf.nn.conv2d(
            pool1,
            self._parameters["conv2_w"],
            [1, 1, 1, 1],
            "SAME",
            name="conv2")  # (?,158,78,32)
        relu2 = tf.nn.leaky_relu(conv2, name="relu2")
        pool2 = tf.nn.max_pool(
            relu2,
            (1, 2, 2, 1),
            (1, 2, 2, 1),
            "SAME",
            name="pool2")  # (?,79,39,32)
        conv3 = tf.nn.conv2d(
            pool2,
            self._parameters["conv3_w"],
            [1, 1, 1, 1],
            "SAME",
            name="conv3")  # (?,78,38,64)
        relu3 = tf.nn.leaky_relu(conv3, name="relu3")
        pool3 = tf.nn.max_pool(
            relu3,
            (1, 2, 2, 1),
            (1, 2, 2, 1),
            "SAME",
            name="pool3")  # (?,39,19,64)
        conv4 = tf.nn.conv2d(
            pool3,
            self._parameters["conv4_w"],
            [1, 1, 1, 1],
            "SAME",
            name="conv4")  # (?,38,18,128)
        relu4 = tf.nn.leaky_relu(conv4, name="relu4")
        pool4 = tf.nn.max_pool(
            relu4,
            (1, 2, 2, 1),
            (1, 2, 2, 1),
            "SAME",
            name="pool4")  # (?,19,9,128)

        reshape1 = tf.concat(
            [tf.reshape(pool4, [-1]), speed, steering_angle], 0)

        relu5 = tf.nn.leaky_relu(
            tf.add(
                tf.matmul(reshape1, self._parameters["relu5_w"]),
                self._parameters["relu5_b"]),
            name="relu5")  # (1024)

        relu6 = tf.nn.leaky_relu(
            tf.add(
                tf.matmul(relu5, self._parameters["relu6_w"]),
                self._parameters["relu6_b"]),
            name="relu6")  # (512)

        relu7 = tf.nn.leaky_relu(
            tf.add(
                tf.matmul(relu6, self._parameters["relu7_w"]),
                self._parameters["relu7_b"]),
            name="relu7")  # (256)

        relu8 = tf.nn.leaky_relu(
            tf.add(
                tf.matmul(relu7, self._parameters["relu8_w"]),
                self._parameters["relu8_b"]),
            name="relu8")  # (256)

        logit = tf.add(
            tf.matmul(relu8, self._parameters["logit_w"]),
            self._parameters["logit_b"])

        return logit # (256) [-128,127]

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
