## Overview
This repo contains the code for learning an RL policy in [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) for a two wheeled inverted pendulum (TWIP) robot. It includes the following folders and subfolders:

1. ```assets```: contains the URDF file of the robot

2. ```code```: contains the IsaacGym environment and the the task specification/PPO parameters

## How to start the learning
1. Download [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

2. Copy the URDF in the assets folder in IsaacGymEnvs/assets

3. Copy the ```Twip.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/task

4. Copy the ```TwipPPO.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/train

5. Copy the ```twip.py``` in the assets folder in IsaacGymEnvs/isaacgymenvs/task

6. run ```python train.py task=Twip```

