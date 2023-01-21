## Overview
This repo contains the code for learning an RL policy in [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) for a two wheeled inverted pendulum (TWIP) and a Flywheel inverted pendulum robots. It includes the following folders and subfolders:

1. ```assets```: contains the URDF file of the robot

2. ```code```: contains the IsaacGym environment and the the task specification/PPO parameters

## How to start the learning
1. Download [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)

2. Copy the URDF file contained in the assets folder in IsaacGymEnvs/assets

3. Copy ```Twip.yaml``` and ```FlywheelPendulum.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/task

4. Copy ```TwipPPO.yaml``` and ```FlywheelPendulumPPO.yaml``` in the folder in IsaacGymEnvs/isaacgymenvs/cfg/train

5. Copy ```twip.py``` and ```flywheel_pendulum.py```in the assets folder in IsaacGymEnvs/isaacgymenvs/task

6. Paste in IsaacGymEnvs/isaacgymenvs/tasks/__init__.py 
```
from .twip import Twip
from .flywheel_pendulum import FlywheelPendulum

....

"Twip": Twip,
"FlywheelPendulum": FlywheelPendulum,
```

7. run ```python train.py task=Twip``` or ```python train.py task=FlywheelPendulum```



