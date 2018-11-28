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

import tensorflow as tf
import numpy as np

from libtrain import DeterministicPolicyGradientAlgorithm as DPG

sio = socketio.Server()
app = Flask(__name__)
algo = None


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

last_image = None
last_speed = 0.0
last_action = 0.0

iter_count = 100
EVAL = False


@sio.on('telemetry')
def telemetry(sid, data):
    global reset_sent
    global start_time
    global RESET_SPEED, max_speed
    global last_timestamp, last_speed
    global last_image, last_speed, last_action
    global iter_count
    global EVAL

    now_timestamp = time.time()
    time_diff = now_timestamp - last_timestamp
    last_timestamp = now_timestamp

    if data:
        # The current steering angle of the car [-25,25]
        steering_angle = float(data["steering_angle"])/25
        # The current throttle of the car [0,1]
        throttle = data["throttle"]
        # The current speed of the car [0,30]
        speed = float(data["speed"])/30
        # The current image from the center camera of the car (320,160,3)
        imgString = data["image"]

        last_reward = time_diff
        total_reward = now_timestamp - start_time
        max_speed = max(speed, max_speed)

        image = Image.open(BytesIO(base64.b64decode(imgString)))

        image_array = np.reshape(np.divide(np.asarray(
            image), 256).astype(np.float32), [160, 320, 3])
        image_array = tf.convert_to_tensor(
            np.transpose(image_array, [1, 0, 2]))

        print('{"last_speed":%f,"last_action":%f,"last_reward":%f,"total_reward":%f,"this_speed":%f}' % (
            last_speed, last_action, last_reward, total_reward, speed))

        if speed < RESET_SPEED and total_reward > 5:
            if not reset_sent:
                send_reset()
                print('{"reset":true}')
                reset_sent = True

        # Control angle [-1,1]
        if not EVAL:
            print('{"evaluation":false}')
            steering_angle, _ = algo.target_actor.inference([[
                tf.reshape(image_array, [-1, 320, 160, 3]),
                tf.convert_to_tensor([[speed]])]])
            value = algo.sess.run(steering_angle)[0][0]
            # steering_angle = random.random()*2
            # print(steering_angle)
            # steering_angle -= 1

            steering_angle = value * 2 - 1 + (random.random()*2 - 1)
            steering_angle = max(steering_angle, -1)
            steering_angle = min(steering_angle, 1)
        else:
            print('{"evaluation":true}')
            steering_angle, _ = algo.target_actor.inference([[
                tf.reshape(image_array, [-1, 320, 160, 3]),
                tf.convert_to_tensor([[speed]])]])
            value = algo.sess.run(steering_angle)[0][0]
            steering_angle = value * 2 - 1

        # Control throttle [0,1]
        throttle = float(1)

        print('{"this_speed":%f,"this_action":%f}' %
              (speed, steering_angle))
        send_control(steering_angle, throttle)

        if last_image is not None:
            if not reset_sent:
                algo.push_buffer(
                    (
                        last_image,
                        [last_speed],
                        [last_action / 2+0.5],
                        [last_reward],
                        image_array,
                        [speed]))
                print("Pushed buffer")

        last_speed = speed
        last_action = steering_angle
        last_image = image_array

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
    global start_time, last_timestamp, last_image, last_speed, last_action, iter_count, reset_sent
    algo.push_buffer(
        (
            last_image,
            [last_speed],
            [last_action / 2+0.5],
            [0.0],
            tf.zeros([320, 160, 3]),
            [0.0]))
    print("Pushed buffer")
    algo.step()
    start_time = time.time()
    last_timestamp = start_time
    last_image = None
    last_speed = 0.0
    last_action = 0.0
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
        algo = DPG(0.5, 0.9, 1024, 0.5, "car_agent", model_checkpoint_path)
        # algo.load()

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
