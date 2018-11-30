import tensorflow as tf
from models.Model import Model
from models.Critic import Critic


class Actor(Model):

    parameter_names = [
        "conv1_w",
        "conv2_w",
        "conv3_w",
        "conv4_w",
        "relu5_w",
        "relu5_b",
        "relu6_w",
        "relu6_b",
        "relu7_w",
        "relu7_b",
        "relu8_w",
        "relu8_b",
        "logit_w",
        "logit_b",
        ]

    def __init__(
            self,
            model_name: str,
            save_path: str):
        self._model_name = model_name
        self._save_path = save_path
        with tf.variable_scope("%s_Actor" % model_name):
            self._parameters = {
                    "conv1_w": tf.get_variable(
                        "conv1_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[3, 3, 4, 32]),
                    "conv2_w": tf.get_variable(
                        "conv2_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[2, 2, 32, 32]),
                    "conv3_w": tf.get_variable(
                        "conv3_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[2, 2, 32, 64]),
                    "conv4_w": tf.get_variable(
                        "conv4_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[2, 2, 64, 64]),
                    "relu5_w": tf.get_variable(
                        "relu5_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[20*10*64 + 1, 512]),
                    "relu5_b": tf.get_variable(
                        "relu5_b",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[512]),
                    "relu6_w": tf.get_variable(
                        "relu6_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[512, 256]),
                    "relu6_b": tf.get_variable(
                        "relu6_b",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[256]),
                    "relu7_w": tf.get_variable(
                        "relu7_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[256, 128]),
                    "relu7_b": tf.get_variable(
                        "relu7_b",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[128]),
                    "relu8_w": tf.get_variable(
                        "relu8_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[128, 128]),
                    "relu8_b": tf.get_variable(
                        "relu8_b",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[128]),
                    "logit_w": tf.get_variable(
                        "logit_w",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[128, 1]),
                    "logit_b": tf.get_variable(
                        "logit_b",
                        initializer=tf.initializers.glorot_normal(),
                        shape=[1])
                }

    def inference(self, X, is_target=False):

        states = X[0]

        states_image = states[0]# (?,320,160,4)
        states_speed = states[1]# (?) [0,1]

        conv1 = tf.nn.conv2d(
            states_image,
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
            [tf.reshape(pool4, [ -1,20*10*64]), states_speed],
            1)

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

        mu = tf.nn.sigmoid(logit, name="mu")

        return mu, logit  # (1) [-128,127]

    def parameters(self):
        return list(self._parameters.values())

    def initialize_parameters(self, sess:tf.Session):
        sess.run(tf.initializers.variables(self.parameters()))

    def save(self, sess: tf.Session):
        saver = tf.train.Saver(
            self.parameters(),
            save_relative_paths=True,
            filename=self._model_name)
        saver.save(sess, "%s/%s_actor"%(self._save_path,self._model_name))

    def load(self, sess: tf.Session):
        saver = tf.train.Saver(
            self.parameters(),
            save_relative_paths=True,
            filename=self._model_name)
        saver.restore(sess, "%s/%s_actor"%(self._save_path,self._model_name))

    def sync(self, target, sess: tf.Session):
        """Sync the parameter value of self to target.
        
        Arguments:
            target {Actor} -- Target of syncing.
        """

        for n in Actor.parameter_names:
            sess.run(
                target._parameters[n].assign(self._parameters[n]))


    def copy(self, model_name: str, save_path: str, sess: tf.Session):
        """Create a new Critic and sync parameter value to it.
        
        Arguments:
            model_name {str} -- Name of new Critic
            save_path {str} -- Save path of new Critic
        
        Returns:
            Actor -- The new Actor model
        """

        new_network = Actor(
            model_name, 
            save_path)

        self.sync(new_network,sess)

        return new_network
