import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import random
import time
import collections

import tensorflow as tf
import numpy as np

from libtrain import DeterministicPolicyGradientAlgorithm as DPG
from libtrain import SupervisedAlgorithm

sio = socketio.Server()
app = Flask(__name__)
algo = None
driver = "manual"
epsilon = 0.5


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 30
controller.set_desired(set_speed)

reset_sent = False

RESET_SPEED = 0.9
max_speed = 0.0

random.seed()

start_time = time.time()
last_timestamp = start_time

last_images = collections.deque(maxlen=4)
last_speed = 0.0
last_action = 0.0

TRAINING = False


def grayscale_pack_motion_image(images):
    images = tf.stack(list(images))
    images = tf.image.rgb_to_grayscale(images)
    images = tf.reshape(images, [4, 320, 160])
    images = tf.transpose(images, [1, 2, 0])
    return images


def random_driver(image_array, speed):
    steering_angle = (random.random() * 2) - 1
    throttle = 1.0
    print('{"action":"random",this_speed":%f,"this_action":%f}' %
          (speed, steering_angle))
    send_control(steering_angle, throttle)


def model_greedy_driver(image_array, speed):
    global algo
    steering_angle, _ = algo.target_actor.inference([[
        tf.reshape(image_array, [-1, 320, 160, 4]),
        tf.convert_to_tensor([[speed]])]])
    value = algo.sess.run(steering_angle)[0][0]
    steering_angle = value * 2 - 1
    throttle = 1.0
    print('{"action":"greedy","this_speed":%f,"this_action":%f}' %
          (speed, steering_angle))
    send_control(steering_angle, throttle)


def model_epsilon_greedy_driver(image_array, speed):
    global epsilon
    if random.random() < epsilon:
        model_greedy_driver(image_array, speed)
    else:
        random_driver(image_array, speed)


def manual_driver(image_array, speed):
    print('{"action":"manual"}')
    sio.emit('manual', data={}, skip_sid=True)


@sio.on('telemetry')
def telemetry(sid, data):
    global reset_sent
    global start_time
    global RESET_SPEED, max_speed
    global last_timestamp, last_speed
    global last_images, last_speed, last_action
    global algo, driver, epsilon
    global TRAINING

    now_timestamp = time.time()
    time_diff = now_timestamp - last_timestamp
    last_timestamp = now_timestamp

    if data:
        # The current steering angle of the car [-25,25]
        steering_angle = float(data["steering_angle"])/50 + 0.5
        # The current throttle of the car [0,1]
        throttle = data["throttle"]
        # The current speed of the car [0,30]
        speed = float(data["speed"])/30
        # The current image from the center camera of the car (320,160,3)
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        image_array = np.reshape(np.divide(np.asarray(
            image), 256).astype(np.float32), [160, 320, 3])
        image_array = tf.convert_to_tensor(
            np.transpose(image_array, [1, 0, 2]))

        last_action = steering_angle
        last_reward = time_diff
        total_reward = now_timestamp - start_time
        max_speed = max(speed, max_speed)

        if not reset_sent:
            print('{"last_speed":%f,"last_action":%f,"last_reward":%f,"total_reward":%f,"this_speed":%f,"max_speed":%f,"reset_speed":%f}' % (
                last_speed, last_action, last_reward, total_reward, speed, max_speed, (RESET_SPEED * max_speed)))

            while len(last_images) != last_images.maxlen:
                last_images.append(image_array)
            packed_last_images = grayscale_pack_motion_image(last_images)
            last_images.append(image_array)
            if speed < (RESET_SPEED * max_speed) and total_reward > 2:
                print('{"reset":true}')
                if TRAINING:
                    if args.model != "supervised":
                        algo.push_buffer(
                            (
                                packed_last_images,
                                [last_speed],
                                [last_action / 2+0.5],
                                [0.0],
                                tf.zeros([320, 160, 4]),
                                [0.0]))
                    print("Pushed buffer")
                    algo.step()
                send_reset()
                reset_sent = True
            else:
                if TRAINING:
                    algo.push_buffer(
                        (
                            packed_last_images,
                            [last_speed],
                            [last_action / 2+0.5],
                            [last_reward],
                            grayscale_pack_motion_image(last_images),
                            [speed]))
                    print("Pushed buffer")

            last_speed = speed

            driver(grayscale_pack_motion_image(last_images), speed)
        else:
            sio.emit('manual', data={}, skip_sid=True)

        # save frame
        if args.record_path != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.record_path, timestamp)
            image.save('{}.jpg'.format(image_filename))
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
    global max_speed, start_time, last_timestamp, last_images, last_speed, last_action, reset_sent

    start_time = time.time()
    last_timestamp = start_time
    last_images = collections.deque(maxlen=4)
    last_speed = 0.0
    last_action = 0.0
    max_speed = 0.0
    reset_sent = False
    print("Reset")


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


def send_reset():
    sio.emit("reset", data={}, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '-m', '--model_path',
        type=str,
        default='',
        help='Path to model checkpoint path.'
    )
    parser.add_argument(
        '-M', '--model',
        type=str,
        default='',
        help='The model used. ["supervised","continuous_deep_q"]'
    )
    parser.add_argument(
        '-d', '--drive',
        type=str,
        default='manual',
        help='The driver used. Default: "manual". ["manual","model_greedy","model_epsilon_greedy","random"]'
    )
    parser.add_argument(
        '-e', '--epsilon',
        type=float,
        default=0.5,
        help='Epsilon as in epsilon-greedy exploration. Default: 0.5.'
    )
    parser.add_argument(
        '-r', '--record_path',
        type=str,
        default='',
        help='Path to recording folder. This is where the images from the run will be saved.'
    )
    parser.add_argument(
        '-l', '--load',
        default=False,
        action="store_true",
        help='Load the model from model_path.'
    )
    parser.add_argument(
        '-t', '--train',
        default=False,
        action="store_true",
        help='Train the model with experience replay.'
    )
    args = parser.parse_args()

    if args.model_path != '':
        model_checkpoint_path = args.model_path
    elif args.model != '':
        print("Argument model_path needed for using a model.")
        exit(1)

    if args.model != '':
        if args.model == 'continuous_deep_q':
            algo = DPG(0.5, 0.9, 1024, 0.5, "car_agent_deep_q",
                       model_checkpoint_path)
        elif args.model == 'supervised':
            algo = SupervisedAlgorithm(
                10000, 128, "car_agent_supervised", model_checkpoint_path)
        else:
            print("Model not recognised.")
            exit(1)
        if args.load:
            algo.load()

    if args.drive == 'manual':
        driver = manual_driver
    elif args.drive == 'model_greedy':
        driver = model_greedy_driver
    elif args.drive == 'model_epsilon_greedy':
        driver = model_epsilon_greedy_driver
    elif args.drive == 'random':
        driver = random_driver
    epsilon = args.epsilon
    TRAINING = args.train

    if args.record_path != '':
        print("Creating image folder at {}".format(args.record_path))
        if not os.path.exists(args.record_path):
            os.makedirs(args.record_path)
        else:
            shutil.rmtree(args.record_path)
            os.makedirs(args.record_path)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
