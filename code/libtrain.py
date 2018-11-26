from models.Critic import Critic
from models.Actor import Actor
import tensorflow as tf
import math
import collections
import random
from debug_printer import debug_printer as dbg

random.seed()


class DeterministicPolicyGradientAlgorithm:
    def __init__(
            self,
            update_weight: float,
            future_reward_discount: float,
            replay_buffer_size: int,
            sample_proportion: float,
            name: str,
            save_path: str,
            action_exploration_distribution=tf.distributions.Normal(
                0.0, 0.2),
            sess: tf.Session = tf.Session()):

        self.future_reward_discount = future_reward_discount
        self.update_weight = update_weight
        self.sess = sess
        self.action_exploration_distribution = action_exploration_distribution
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
        self.save()

    def critic_train_y(self, transitions: list, batch_size: int):

        rewards = transitions[2]
        next_states = transitions[3]

        if next_states is None:
            y = tf.zeros((batch_size,))
        else:
            next_target_action = self.target_actor.inference([next_states])
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

        actor_value = self.actor.inference([states])

        critic_value = self.critic.inference([states, actor_value])

        loss = -critic_value

        return loss

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

    def optimize(
            self,
            transitions: list,
            batch_size: int,
            optimizer=tf.train.GradientDescentOptimizer(0.00001)):

        self.sess.run(optimizer.minimize(
            self.critic_loss(transitions, batch_size),
            var_list=self.critic.parameters()))

        self.sess.run(optimizer.minimize(
            self.actor_loss(transitions, batch_size),
            var_list=self.actor.parameters()))

    def exploration_action(self, states):
        action_exploration = self.actor.inference(
            [states]) + self.action_exploration_distribution.sample()
        values = self.sess.run(action_exploration)
        return values

    def push_buffer(self, transition: tuple):
        self.replay_buffer.append(transition)

    def step(self, transition: tuple):
        dbg()
        self.push_buffer(transition)
        sample_size = math.ceil(
            self.sample_proportion * len(self.replay_buffer))
        sample = [self.replay_buffer[i]
                  for i in random.sample(
                      range(len(self.replay_buffer)),
                      sample_size)]

        dbg()
        states = []
        actions = []
        rewards = []
        next_states = []
        for t in sample:
            states.append((t[0][0], t[0][1]))
            actions.append(t[1])
            rewards.append(t[2])
            next_states.append(t[3])
        transitions = (states, actions, rewards, next_states)
        dbg()
        self.optimize(transitions, len(sample))
        dbg()
        self.update_target()
        dbg()
        self.save()
        dbg()

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
