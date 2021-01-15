#!/bin/bash

# initialize python virtual environment
source ~/python-venv/cgan/bin/activate

cd server

# start serving app
python main.py

# leave python virtual environment
deactivate


