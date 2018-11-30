import argparse
import base64
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import numpy as np
import random
import time
from collections import deque
import tensorflow as tf
from skimage import transform
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.device('/device:GPU:0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


sio = socketio.Server()
app = Flask(__name__)
model = None
sess = None

RESET_SPEED_DIFF = -0.1
NO_RESET_PERIOD = 1
start_time = 0
speed = 0.0
stack_size = 4
state = np.zeros((84, 84, 4), dtype=float)
stacked_frames = deque([np.zeros((84, 84), dtype=float) for i in range(stack_size)], maxlen=4)
throttle = 1.
loss = 0.

total_episodes = 100        # Total episodes for training
max_steps = 100000000              # Max possible steps in an episode
batch_size = 64
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time

memory_size = 1000000          # Number of experiences the Memory can keep

done = False
evaluate = False
train = False
is_pretrain_finished = False
is_episode_finished = True
start_new_evaluation = True

possible_actions = [-0.7, -0.3, 0.0, 0.0, 0.0, 0.3, 0.7]
action = 0.0

# Q learning hyperparameters
gamma = 0.95  # Discounting rate
step = 0
# Initialize the decay rate (that will use to reduce epsilon)
decay_step = 0
episode = 0

# Initialize the rewards of the episode
episode_rewards = []
explore_probability_ = 0.
exp_exp_tradeoff = 1.


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size_):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size_,
                                 replace=False)

        return [self.buffer[i] for i in index]


memory = Memory(max_size=memory_size)


def send_control(steering_angle, throttle_):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle_.__str__()
        },
        skip_sid=True)


def send_reset():
    sio.emit("reset", data={}, skip_sid=True)


def preprocess_frame(imgString):
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_gray = image.convert('L')
    image_array = np.asarray(image_gray)
    cropped_frame = image_array[40:, :]
    normalized_frame = cropped_frame/255.0
    frame = transform.resize(normalized_frame, [84, 84])
    return frame


def stack_frames(stacked, state_, is_new_episode):
    frame = preprocess_frame(state_)
    if is_new_episode:
        stacked = deque([np.zeros((84, 84), float) for i in range(stack_size)], maxlen=4)

        stacked.append(frame)
        stacked.append(frame)
        stacked.append(frame)
        stacked.append(frame)

        stacked_state = np.stack(stacked, axis=2)
    else:
        stacked.append(frame)
        stacked_state = np.stack(stacked, axis=2)
    return stacked_state, stacked


class DQNetwork:
    def __init__(self, model_name: str):
        self.state_size = [84, 84, 4]
        self.action_size = 7
        self.learning_rate = 0.0002
        self._model_name = model_name

        with tf.variable_scope(self._model_name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *self.state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 7], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            # --> [20, 20, 32]

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            # --> [9, 9, 64]

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            # --> [1152]

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


def sparse_action(action_):
    x = np.eye(7)
    a_0, a_1, a_2, a_3, a_4, a_5, a_6 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    actions = [a_0, a_1, a_2, a_3, a_4, a_5, a_6]
    sparsed_action = a_3
    for i in range(7):
        if action_ == possible_actions[i]:
            sparsed_action = actions[i]
    return sparsed_action


def predict_action(decay_step_, _state):
    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.0001  # exponential decay rate for exploration prob
    # EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step_)

    if explore_probability > exp_exp_tradeoff:
        # Make a random action (exploration)
        _action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(model.output, feed_dict={model.inputs_: _state.reshape((1, *_state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        _action = possible_actions[int(choice)]

    return _action, explore_probability


@sio.on('telemetry')
def telemetry(sid, data):
    global speed
    global model
    global sess
    global memory
    global stacked_frames
    global state
    global action
    global is_pretrain_finished, done, is_episode_finished, start_new_evaluation
    global step, decay_step, episode
    global train, evaluate
    global episode_rewards
    global explore_probability_
    global loss

    if data:
        if model is None:
            next_speed = float(data["speed"])
            speed_diff = next_speed - speed
            speed = next_speed
            # frame = preprocess_frame(data["image"])
            # plt.imshow(frame, cmap='gray')
            # plt.show()
            if time.time() - start_time > NO_RESET_PERIOD:
                if speed_diff < RESET_SPEED_DIFF:
                    send_reset()
            # action
            # Make a random action (exploration)
            steering_angle = random.choice(possible_actions)
            send_control(steering_angle, throttle)
        elif train:
            # pre-train
            if not is_pretrain_finished:
                if step == 0:
                    done = False
                    frame = data["image"]
                    state, stacked_frames = stack_frames(stacked_frames, frame, True)
                    speed = float(data["speed"])
                    # Make a random action (exploration)
                    action = random.choice(possible_actions)
                    send_control(action, throttle)
                else:
                    next_speed = float(data["speed"])
                    reward = next_speed - speed
                    if (time.time() - start_time > NO_RESET_PERIOD and reward < RESET_SPEED_DIFF) \
                            or step == pretrain_length:
                        done = True
                        next_state = np.zeros(state.shape)
                        sparsed_action = sparse_action(action)
                        memory.add((state, sparsed_action, reward, next_state, done))
                        send_reset()
                        frame = data["image"]
                        state, stacked_frames = stack_frames(stacked_frames, frame, True)
                    else:
                        frame = data["image"]
                        next_state, stacked_frames = stack_frames(stacked_frames, frame, False)
                        # Add experience to memory
                        sparsed_action = sparse_action(action)
                        memory.add((state, sparsed_action, reward, next_state, done))
                        state = next_state
                    speed = next_speed
                    # Make a random action (exploration)
                    action = random.choice(possible_actions)
                    send_control(action, throttle)
                if step == pretrain_length:
                    is_pretrain_finished = True
                step += 1
            # train
            else:
                if episode == 0:
                    #     # Initialize the variables
                    #     sess.run(tf.global_variables_initializer())
                    # else:
                    saver.restore(sess, model_checkpoint_path)
                # episodes
                if episode < total_episodes:
                    if is_episode_finished:
                        # LEARNING PART
                        # Obtain random mini-batch from memory
                        batch = memory.sample(batch_size)
                        states_mb = np.array([each[0] for each in batch], ndmin=3)
                        actions_mb = np.array([each[1] for each in batch])
                        rewards_mb = np.array([each[2] for each in batch])
                        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                        dones_mb = np.array([each[4] for each in batch])

                        target_Qs_batch = []
                        # Get Q values for next_state
                        Qs_next_state = sess.run(model.output, feed_dict={model.inputs_: next_states_mb})
                        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r+gamma*maxQ(s', a')
                        for i in range(len(batch)):
                            terminal = dones_mb[i]
                            # If we are in a terminal state, only equals reward
                            if terminal:
                                target_Qs_batch.append(rewards_mb[i])
                            else:
                                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                                target_Qs_batch.append(target)
                        targets_mb = np.array([each for each in target_Qs_batch])
                        loss, _ = sess.run([model.loss, model.optimizer],
                                           feed_dict={model.inputs_: states_mb,
                                                      model.target_Q: targets_mb,
                                                      model.actions_: actions_mb})
                        # Write TF Summaries
                        summary = sess.run(write_op, feed_dict={model.inputs_: states_mb,
                                                                model.target_Q: targets_mb,
                                                                model.actions_: actions_mb})
                        writer.add_summary(summary, episode)
                        writer.flush()
                        print('Model Updated')

                        episode += 1
                        if episode % 100 == 0:
                            # Save model every episode
                            saver.save(sess, model_checkpoint_path)
                            print("Model Saved")

                        episode_rewards = []
                        done = False
                        step = 0
                        is_episode_finished = False
                        # observe the first state
                        frame = data["image"]
                        state, stacked_frames = stack_frames(stacked_frames, frame, True)
                        speed = float(data["speed"])
                        # Predict the action to take and take it
                        action, explore_probability_ = predict_action(decay_step, state)
                        send_control(action, throttle)
                        step += 1
                    else:
                        step += 1
                        # Increase decay_step
                        decay_step += 1
                        next_speed = float(data["speed"])
                        reward = next_speed - speed
                        # Add the reward to total reward
                        episode_rewards.append(reward)
                        if (time.time() - start_time > NO_RESET_PERIOD and reward < RESET_SPEED_DIFF) \
                                or step == max_steps:
                            send_reset()
                            done = True
                            # We finished the episode
                            next_state = np.zeros(state.shape)
                            # Add experience to memory
                            sparsed_action = sparse_action(action)
                            memory.add((state, sparsed_action, reward, next_state, done))
                            # end the episode
                            is_episode_finished = True
                            # Get the total reward of the episode
                            total_reward = np.sum(episode_rewards)
                            print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Training loss: {:.4f}'.format(loss),
                                  'Explore P: {:.4f}'.format(explore_probability_))
                        else:
                            frame = data["image"]
                            next_state, stacked_frames = stack_frames(stacked_frames, frame, False)
                            # Add experience to memory
                            sparsed_action = sparse_action(action)
                            memory.add((state, sparsed_action, reward, next_state, done))
                            state = next_state
                            speed = next_speed
                            action, explore_probability_ = predict_action(decay_step, state)
                            send_control(action, throttle)
        elif evaluate:
            if start_new_evaluation:
                # Load the model
                saver.restore(sess, model_checkpoint_path)
                start_new_evaluation = False
                state, stacked_frames = stack_frames(stacked_frames, data["image"], True)
                speed = float(data["speed"])
                Qs = sess.run(model.output, feed_dict={model.inputs_: state.reshape((1, *state.shape))})
                action_ = np.argmax(Qs)
                action = possible_actions[int(action_)]
                send_control(action, throttle)
            else:
                next_speed = float(data["speed"])
                reward = next_speed - speed
                if time.time() - start_time > NO_RESET_PERIOD and reward < RESET_SPEED_DIFF:
                    send_reset()
                    start_new_evaluation = True
                else:
                    state, stacked_frames = stack_frames(stacked_frames, data["image"], False)
                    speed = next_speed
                    # Take the biggest Q value (= the best action)
                    Qs = sess.run(model.output, feed_dict={model.inputs_: state.reshape((1, *state.shape))})
                    action_ = np.argmax(Qs)
                    action = possible_actions[int(action_)]
                    send_control(action, throttle)
    else:
            # NOTE: DON'T EDIT THIS.
            sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    global start_time
    start_time = time.time()
    print("connect ", sid)
    send_control(0, 0)


@sio.on('reset')
def reset(sid, environ):
    print("Reset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='./models/model.ckpt',
        help='Path to model checkpoint path.'
    )
    parser.add_argument(
        'task',
        type=str,
        nargs='?',
        default='evaluate',
        help='train or evaluate'
    )
    args = parser.parse_args()

    if args.model != '':
        tf.reset_default_graph()
        model_checkpoint_path = args.model
        model = DQNetwork('CarAgent')
        saver = tf.train.Saver()
        sess = tf.Session(config=config)
        if args.task == 'train':
            train = True
            # Setup TensorBoard Writer
            writer = tf.summary.FileWriter("./tensorboard/dqn/1")
            # Losses
            tf.summary.scalar("Loss", model.loss)
            write_op = tf.summary.merge_all()

        elif args.task == 'evaluate':
            evaluate = True

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)



