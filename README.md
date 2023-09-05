# Heron
This is the implementation of HERON, a method for reward design from weak reward signals. This repository contains code for four different experiments: Classic Control, Robotics, Code Generation, and Multi-Agent Traffic Light Control.

## Installation
For classic control, run the following commands:
```
pip install -r ClassicControl/requirements/requirements.txt
pip install -e gym
```

For robotics and code generation, follow the instructions in the respective folders.

## Running
To run experiments in the Classic Control settings, run the following commands
```
cd ClassicControl
bash mountain.sh
bash pendulum.sh
```

To run experiments in the Robotics environments, run
```
cd robots
bash run_files/ant.sh
```