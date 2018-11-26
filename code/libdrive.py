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

from libtrain import DeterministicPolicyGradientAlgorithm
import tensorflow as tf
import numpy as np

sio = socketio.Server()
app = Flask(__name__)
algo = None
prev_image_array = None


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

reset_sent = True

RESET_SPEED = 0.8
RESET_SPEED_DIFF = -0.4

RESET_BY_SPEED = True
RESET_BY_SPEED_DIFF = True

NO_RESET_PERIOD = 5
start_time = time.time()

random.seed()
RANDOM_ACTION_PROB = 0.5

algo = None
sess = None
last_timestamp = start_time
last_speed = 0.0

last_state = None
last_action = None


@sio.on('telemetry')
def telemetry(sid, data):
    global reset_sent
    global RESET_SPEED, RESET_SPEED_DIFF
    global RESET_BY_SPEED, RESET_BY_SPEED_DIFF
    global RANDOM_ACTION_PROB
    global last_timestamp, last_speed
    global algo, sess
    global NO_RESET_PERIOD, start_time
    global last_state, last_action

    now_timestamp = time.time()
    time_diff = now_timestamp - last_timestamp
    last_timestamp = now_timestamp
    time_elasped = now_timestamp - start_time

    if data:
        # The current steering angle of the car [-25,25]
        steering_angle = float(data["steering_angle"])/25
        # The current throttle of the car [0,1]
        throttle = data["throttle"]
        # The current speed of the car [0,30]
        speed = float(data["speed"])/30
        # The current image from the center camera of the car (320,160,3)
        imgString = data["image"]

        speed_diff = (speed - last_speed) / time_diff
        last_reward = time_diff
        last_speed = speed

        print('{"steering_angle":%f,"speed":%f,"time_diff":%f,"time_elasped":%f}' % (
            steering_angle,
            speed,
            time_diff,
            time_elasped))

        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.divide(np.asarray(image), 256).astype(np.float32)
        image_array = tf.convert_to_tensor(image_array)
        image_array = tf.transpose(image_array, [1, 0, 2])
        image_array = tf.reshape(image_array, [-1, 320, 160, 3])
        tfspeed = tf.convert_to_tensor([[speed]])
        state = (image_array, tfspeed)

        if time.time() - start_time > NO_RESET_PERIOD:
            if RESET_BY_SPEED_DIFF:
                if speed_diff < RESET_SPEED_DIFF:
                    if not reset_sent:
                        send_reset()
                        if last_state is not None:
                            if algo is not None:
                                algo.step(
                                    (
                                        last_state,
                                        last_action,
                                        0,
                                        None))
                            print('{"DPG":"step","reset":true}')
                        start_time = time.time()
                        last_state = None
                        last_action = None
                        reset_sent = True
                else:
                    reset_sent = False

            if RESET_BY_SPEED:
                if speed < RESET_SPEED:
                    if not reset_sent:
                        send_reset()
                        if last_state is not None:
                            if algo is not None:
                                algo.step(
                                    (
                                        last_state,
                                        last_action,
                                        0,
                                        None))
                            print('{"DPG":"step","reset":true}')
                        start_time = time.time()
                        last_state = None
                        last_action = None
                        reset_sent = True
                else:
                    reset_sent = False

        if last_state is not None:
            if algo is not None and not reset_sent:
                algo.step((last_state, last_action, last_reward, state))
                print('{"DPG":"step","reset":false}')
        last_state = state

        # Control angle [-1,1]
        action = algo.exploration_action(state)
        steering_angle = action
        steering_angle = steering_angle * 2 - 1
        last_action = steering_angle

        # Control throttle [0,1]
        throttle = float(1)

        send_control(steering_angle, throttle)
        print('{"control":{"steering_angle":%f,"throttle":%f}}' %
              (steering_angle, throttle))

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
        'model',
        type=str,
        nargs='?',
        default='',
        help='Path to model checkpoint path.'
    )
    parser.add_argument(
        'record_path',
        type=str,
        nargs='?',
        default='',
        help='Path to recording folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    if args.model != '':
        model_checkpoint_path = args.model
        algo = DeterministicPolicyGradientAlgorithm(
            0.5,
            0.9,
            128,
            0.6,
            "car_agent",
            model_checkpoint_path)
        sess = algo.sess
        algo.load()

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
