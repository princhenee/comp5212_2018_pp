from models.Critic import Critic
from models.Actor import Actor
import tensorflow as tf


class DeterministicPolicyGradientAlgorithm:
    def __init__(
            self,
            update_weight: float,
            future_reward_discount: float,
            name: str,
            save_path: str,
            action_exploration_distribution=tf.distributions.Normal(
                0, 0.2),
            sess: tf.Session = tf.Session()):

        self.future_reward_discount = future_reward_discount
        self.update_weight = update_weight
        self.sess = sess

        self.critic = Critic(name, save_path)
        self.critic.initialize_parameters(sess)
        self.target_critic = self.critic.copy(
            "%s_target" % name, save_path, sess)

        self.actor = Actor(name, save_path)
        self.actor.initialize_parameters(sess)
        self.target_actor = self.actor.copy("%s_target", save_path, sess)

    def critic_train_y(self, transitions: list):

        rewards = transitions[2]
        next_states = transitions[3]

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

    def critic_loss(self, transitions: list):

        states = transitions[0]
        actions = transitions[1]

        y = self.critic_train_y(transitions)

        critic_value = self.critic.inference([states, actions])

        loss = tf.losses.mean_squared_error(y, critic_value)

        return loss

    def actor_loss(self, transitions: list):

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
