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

# from models.QValueModel import QValueModel
from models.DeepQModel import DeepQAgent
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
# parameters to cut images
IMAGE_X_LOWER = 60
IMAGE_X_UPPER = 140
IMAGE_Y_SIZE = 360
STATE_SIZE = (IMAGE_X_UPPER - IMAGE_X_LOWER, IMAGE_Y_SIZE)

def image_preprocess(image_array):
    # original input is RGB figure
    # using the RED channel only
    image_input = image_array[IMAGE_X_LOWER: IMAGE_X_UPPER, :, 0]
    return image_input

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
start_time = 0

random.seed()
RANDOM_ACTION_PROB = 0.5

model = None
sess = None
last_timestamp = 0.0
last_speed = 0.0


@sio.on('telemetry')
def telemetry(sid, data):
    global reset_sent
    global RESET_SPEED, RESET_SPEED_DIFF
    global RESET_BY_SPEED, RESET_BY_SPEED_DIFF
    global RANDOM_ACTION_PROB
    global last_timestamp, last_speed
    global model, sess
    global NO_RESET_PERIOD, start_time

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
        print imgString.shape

        speed_diff = (speed - last_speed) / time_diff
        last_reward = speed - last_speed
        last_speed = speed

        # action discretisize
        action_size = 11
        # steering angle range
        steering_angle_UPPER = 1.0
        steering_angle_LOWER = -1.0
        def get_steering_angle_by_action(action):
            return np.linspace(steering_angle_LOWER,
                               steering_angle_UPPER,
                               action_size)[action]
                                            
        # iterations #
        episodes = 100
        for time_t in range(NO_RESET_PERIOD):
            action = 0


        print("Feedback:", steering_angle, throttle, speed, speed_diff)

        if time.time() - start_time > NO_RESET_PERIOD:
            if RESET_BY_SPEED_DIFF:
                if speed_diff < RESET_SPEED_DIFF:
                    if not reset_sent:
                        send_reset()
                        reset_sent = True
                else:
                    reset_sent = False

            if RESET_BY_SPEED:
                if speed < RESET_SPEED:
                    if not reset_sent:
                        send_reset()
                        reset_sent = True
                else:
                    reset_sent = False

        image = Image.open(BytesIO(base64.b64decode(imgString)))

        image_array = np.asarray(image)
        image_input = image_preprocess(image_array)

        # Control throttle [0,1]
        throttle = float(0)

        print("Control:", steering_angle, throttle)
        print()
        send_control(steering_angle, throttle)

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

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
