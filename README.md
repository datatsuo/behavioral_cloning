# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
The goal of this project is to clone the behavior of the car
in the simulator by building a convolutional neural network (CNN) model.

## Contents of Repository
The followings are the contents of this repository
(see also the repository for this project by Udacity
[here](https://github.com/udacity/CarND-Behavioral-Cloning-P3)):

- `model.py`: Python code for our CNN model for driving the car.
- `model.h5`: Our trained model is stored
- `drive.py`: Python code for driving the car (run `python drive.py model.h5`)
before starting the simulator in the autonomous mode (unchanged from the one
provided by Udacity
[here](https://github.com/udacity/CarND-Behavioral-Cloning-P3)).
- `writeup_report.md`: summary for this project
- `Behavioral_Cloning.ipynb`: Jupyter notebook for this project (essentially
  the same as `model.py` but some codes for visualization and are included).
- `Behavioral_Cloning.html`: Html export of `Behavioral_Cloning.ipynb`.
- `videos`: this folder includes the videos of the car with our model running
on the track 1 and 2 in the simulator.
- `figures`: this folder includes images used in the `writeup_report.md`.
- `video.py`: this python code is used for creating the file (see the repository
  by Udacity for more detail.)
- `requirements.txt`: a list of libraries used in `model.py`.

The raw training/validation data for this project are available at
[track1](https://www.dropbox.com/s/dvqycnpnfou97nv/data.zip?dl=0)
and
[track2](https://www.dropbox.com/s/jtid1xlprml1d7s/data2.zip?dl=0).
