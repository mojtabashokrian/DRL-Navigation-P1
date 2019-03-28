# DRL-Navigation-P1
Project 1 Udacity's Deep RL nanodegree

##### &nbsp;

## Goal
In this project, we train a reinforcement learning (RL) agent that navigates an environment similar to [Unity's Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). The environment has yellow and blue bananas scattered around a in rectangle-shaped environment. Occasionally some bananas drop from the air. The environment is episodic and runs for 300 timesteps (30 seconds).

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal is to collect as many yellow bananas as possible while avoiding blue bananas. The environment is solved when the agent achieves an average score of +13 over 100 consecutive episodes.

##### &nbsp;

## State & Action Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction (check the detailed code [here](https://github.com/Unity-Technologies/ml-agents/blob/e82450ab8304093871fd19b876a0f819d390e79d/UnitySDK/Assets/ML-Agents/Examples/SharedAssets/Scripts/RayPerception.cs#L11)). Four discrete actions are available:

- `0` move forward
- `1` move backward
- `2` turn left
- `3` turn right

##### &nbsp;

Read the `Report.md` for full implementation details.

## Report Contents

1. Goal, State and Action Space.
2. Training a vanilla DQN.
3. Visualizing Results and Going Further: The problem of Instability/Loops
4. Learnable Discount Function
5. Time Awareness 
6. Possible Future Improvements and Directions

---

# Project Starter Code
The project starter code can be found below, in case you want to run this project yourself.

Also, the original Udacity repo for this project can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.

### Instructions
  Follow the instructions in `Train.ipynb` to train the agent.
