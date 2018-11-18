# Project codes

## Setup developing environment

1. Make sure you have the following dependencies installed:

```
python3
pip3
virtualenv
```

2. Run `start-develop.sh`

3. Everytime you need to develop, run `. .env`

4. After you are done developing, run `. .out`

## Machine Learning Model

- Model input: {800x600x3 image, speed, throttle, steer_angle} the state of the car
- Model output: {-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10} the change in steering angle

## Car Stimulator Drive (`drive.py`)

- Keep constant (max) throttle.
- Receive state of car from car simulator telemetry.
- Calculate change in speed.
- Model inference, instruct car stimulator to take new steering angle.
- Reset car stimulator when change in speed is negative.
- Record (car_state,model_output,next_change_in_speed) tuple for training.
