from models.Critic import Critic
from models.Actor import Actor
import tensorflow as tf
import math
import collections
import random
from debug import debug_timer


class DeterministicPolicyGradientAlgorithm:
    def __init__(
            self,
            update_weight: float,
            future_reward_discount: float,
            replay_buffer_size: int,
            sample_proportion: float,
            name: str,
            save_path: str,
            sess: tf.Session = tf.Session()):

        self.optimizer1 = tf.train.GradientDescentOptimizer(0.001)
        self.optimizer2 = tf.train.GradientDescentOptimizer(0.001)

        sess.run(tf.initializers.global_variables())
        sess.run(tf.initializers.local_variables())

        self.future_reward_discount = future_reward_discount
        self.update_weight = update_weight
        self.sess = sess
        self.replay_buffer = collections.deque(maxlen=replay_buffer_size)
        self.sample_proportion = sample_proportion

        self.critic = Critic(name, save_path)
        self.critic.initialize_parameters(sess)
        self.target_critic = self.critic.copy(
            "%s_target" % name, save_path, sess)

        self.actor = Actor(name, save_path)
        self.actor.initialize_parameters(sess)
        self.target_actor = self.actor.copy(
            "%s_target" % name, save_path, sess)

    def critic_train_y(self, transitions: list, batch_size: int):

        rewards = transitions[2]
        next_states = transitions[3]

        next_target_action, _ = self.target_actor.inference([next_states])
        next_target_critic = self.target_critic.inference(
            [next_states, next_target_action])

        y = tf.add(
            tf.scalar_mul(
                self.future_reward_discount,
                next_target_critic
            ),
            rewards
        )

        return y

    def critic_loss(self, transitions: list, batch_size: int):

        states = transitions[0]
        actions = transitions[1]

        y = self.critic_train_y(transitions, batch_size)

        critic_value = self.critic.inference([states, actions])

        loss = tf.losses.mean_squared_error(y, critic_value)

        return loss

    def actor_loss(self, transitions: list, batch_size: int):

        states = transitions[0]

        actor_value, _ = self.actor.inference([states])

        critic_value = self.critic.inference([states, actor_value])

        loss = -critic_value

        return loss

    @debug_timer
    def update_target(self):

        for n in Critic.parameter_names:

            target = tf.scalar_mul(
                self.update_weight,
                self.critic._parameters[n])
            target += tf.scalar_mul(
                (1-self.update_weight),
                self.target_critic._parameters[n])

            self.sess.run(self.target_critic._parameters[n].assign(target))

        for n in Actor.parameter_names:

            target = tf.scalar_mul(
                self.update_weight,
                self.actor._parameters[n])
            target += tf.scalar_mul(
                (1-self.update_weight),
                self.target_actor._parameters[n])

            self.sess.run(self.target_actor._parameters[n].assign(target))

    @debug_timer
    def optimize(
            self,
            transitions: list,
            batch_size: int,
            optimizer1=None,
            optimizer2=None):
        if optimizer1 is None:
            optimizer1 = self.optimizer1
        if optimizer2 is None:
            optimizer2 = self.optimizer2
        self.sess.run(optimizer1.minimize(
            self.critic_loss(transitions, batch_size),
            var_list=self.critic.parameters()))

        self.sess.run(optimizer2.minimize(
            self.actor_loss(transitions, batch_size),
            var_list=self.actor.parameters()))

    def push_buffer(self, transition: tuple):
        self.replay_buffer.append(transition)

    @debug_timer
    def step(self):
        sample_size = math.ceil(
            self.sample_proportion * len(self.replay_buffer))
        sample = [self.replay_buffer[i]
                  for i in random.sample(
                      range(len(self.replay_buffer)),
                      sample_size)]

        image = []
        speed = []
        actions = []
        rewards = []
        next_image = []
        next_speed = []
        for t in sample:
            image.append(t[0])
            speed.append(t[1])
            actions.append(t[2])
            rewards.append(t[3])
            next_image.append(t[4])
            next_speed.append(t[5])

        image = tf.stack(image)
        speed = tf.convert_to_tensor(speed)
        actions = tf.convert_to_tensor(actions)
        next_image = tf.stack(next_image)
        next_speed = tf.convert_to_tensor(next_speed)
        transitions = (
            (image, speed),
            actions,
            rewards,
            (next_image, next_speed))
        self.optimize(transitions, len(sample))
        self.update_target()
        self.save()

    def save(self):
        self.critic.save(self.sess)
        self.target_critic.save(self.sess)
        self.actor.save(self.sess)
        self.target_actor.save(self.sess)

    def load(self):
        self.critic.load(self.sess)
        self.target_critic.load(self.sess)
        self.actor.load(self.sess)
        self.target_actor.load(self.sess)


class SupervisedAlgorithm:
    def __init__(
            self,
            name: str,
            save_path: str,
            sess: tf.Session = tf.Session()):

        self.optimizer = tf.train.AdamOptimizer()

        sess.run(tf.initializers.global_variables())
        sess.run(tf.initializers.local_variables())

        self.sess = sess

        self.target_actor = Actor(name, save_path)
        self.target_actor.initialize_parameters(sess)

    def actor_loss(self, transitions: list, batch_size: int):

        states = transitions[0]
        actions = transitions[1]

        _, actor_logits = self.target_actor.inference([states])

        y = actions

        loss = tf.losses.sigmoid_cross_entropy(y, actor_logits)

        return loss

    @debug_timer
    def optimize(
            self,
            transitions: list,
            batch_size: int,
            optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        self.sess.run(optimizer.minimize(
            self.actor_loss(transitions, batch_size),
            var_list=self.target_actor.parameters()))

    @debug_timer
    def step(self, transitions: list):

        image = []
        speed = []
        actions = []
        rewards = []
        next_image = []
        next_speed = []
        for t in transitions:
            image.append(t[0])
            speed.append(t[1])
            actions.append(t[2])
            rewards.append(t[3])
            next_image.append(t[4])
            next_speed.append(t[5])

        image = tf.stack(image)
        speed = tf.convert_to_tensor(speed)
        actions = tf.convert_to_tensor(actions)
        next_image = tf.stack(next_image)
        next_speed = tf.convert_to_tensor(next_speed)
        transitions = (
            (image, speed),
            actions,
            rewards,
            (next_image, next_speed))
        self.optimize(transitions, len(transitions))
        self.save()

    def save(self):
        self.target_actor.save(self.sess)

    def load(self):
        self.target_actor.load(self.sess)
