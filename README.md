# DRL-Navigation-P1
Project 1 Udacity's Deep RL nanodegree

##### &nbsp;

## Goal
In this project, we train a reinforcement learning (RL) agent that navigates an environment similar to [Unity's Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). The environment has yellow and blue bananas scattered around in rectangle shaped environment. Occasionally some bananas drop from the air. The environment is episodic and runs for 300 timesteps (30 seconds).

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal is to collect as many yellow bananas as possible while avoiding blue bananas. The environment is solved when the agent achieves an average score of +13 over 100 consecutive episodes.

##### &nbsp;

## Contents

1. State and Action Space.
2. Training a vanilla DQN.
3. Visualizing Results and Going Further: The problem of Instability/Loops
4. Learnable Discount Factor
5. Select best performing agent and capture video of it navigating the environment.

##### &nbsp;

### 1. State & Action Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction (check the detailed code [here](https://github.com/Unity-Technologies/ml-agents/blob/e82450ab8304093871fd19b876a0f819d390e79d/UnitySDK/Assets/ML-Agents/Examples/SharedAssets/Scripts/RayPerception.cs#L11)). Four discrete actions are available:

- `0` move forward
- `1` move backward
- `2` turn left
- `3` turn right


##### &nbsp;

### 2. Training a vanilla DQN
We first train our agent using a vanilla DQN with two fully connected layers of size 32. We note that all our methods in this project use a __Replay Buffer__ which provides the dataset and sampling for *experience replay*.  With some search in hyperparameters space (see `hyperparams.py`), we solve the environment in 225 episodes, or 125, if following the practice of ignoring the first 100 episodes.


ATTACH FILES WITH DESCRIPTION


##### &nbsp;

### 3. Visualizing Results and Going Further: The Problem of Instability/Loops
We train the agents and observe its behavior in many episodes. Here is an example of a normal run:


But we find out that although the agent has technically solved the environment, there is instability in the results; a typical problem of DRL agents which results in their unreliability and is an obstacle to practical implementation, see [here](https://www.alexirpan.com/2018/02/14/rl-hard.html) for a long post on the problems of deep RL, esp. the section on **The Final Results Can be Unstable and Hard to Reproduce**. The significant drops (sometimes to *zero*) happens when the agent gets stuck in a sort of a loop of actions. Here are many examples of such behavior:

ATTACH FILES WITH DESCRIPTION


We would like to count how many times we get stuck in such loops. It is not easy to *exactly* evaluate the number of loops. When the agent gets stuck in a loop, especially if it is with a *short* period which is mostly the case, it is seeing the exact same states many times (the seen part of the environment is very unlikely to change). Hence we could count if any state has happened for more than `counter_states` time in an episode. If this happens, we count that as a loop. In fact, this is the very mechanism used further below to detect the episodes with loops as we recorded 2 hours of agent's play. We see that in 200 episodes we have *this many* loops.


This could partially go to the deep problem of exploration/exploitation dilemma or simply to the problem of the lack of training, as 225 episodes may not have been enough yet to adequately update the action value across the environment space. So we first train further for a couple of hundred more episodes to reach score 16. Watching that agent play, reveals far fewer loops in 200 episodes. Notice that some drops in the score could also be due to the sparsity of the yellow bananas in that particular episode.

ATTACH FILES WITH DESCRIPTION

Still we can see loops happening sometimes right at the beginning. In the examples below, we have tried to implement also a `loop_breaker` mechanism where we simply make the agent choose a random action with probability 0.99 if it has observed same state more than `counter_states=3`. It has varying degree of success. 


ATTACH FILES WITH DESCRIPTION


##### &nbsp;

### 4. Learnable Discount Factor

In the following two sections, we will discuss strategies that are _conjectured_ to improve a DQN performance. As we are only analyzing one rather simple game here, and the episode length is rather small (300 seconds), the vanilla DQN does a similar job as the other more sophisticated implementations below. Therefore, consider this next two sections as ideas that could provide improvements in more complicated environments.

Perhaps the most fundamental parameter in the basic RL framework which regulates the behavior of the agent is the discount factor `gamma`; ranging from 0 to 1, it provides a certain *spectrum* of possible strategies. The question is, given the current problem, what is the best discount factor, for which the derived optimal policy would really be achieving the ultimate goal: Maximizing the **total** rewards in 300 timesteps. Imagine a course of action in which we collect 10 bananas in the first ten timesteps and another in which we collect 11 bananas in the last 11 timesteps.

As mentioned the **total** rewards (_undiscounted_) is our *criteria*. Hence, obviously the latter is the better course of action but an exponential weighting lesser than 1 like `gamma=0.999` gives a higher score to the _former_ course of actions. Therefore, a policy optimized with `gamma=1` is generally better than any other policy coming from a discount factor less than 1. 

It should be noted that in our case where we only have 300 timesteps, we *can* implement this discount factor and we will get similar results. But as mentioned at the beginning of this section, these are ideas that are conjectured to generally improve a DQN performance for more complicated tasks.

It is easy to see (as we also see it in real life examples) that coming up with a shortterm reward based strategy is much easier than a longterm reward based strategy, and this is related to the fact that the Q-value function to be approximated for shorterm lookahead strategies (small values of `gamma`) is much simpler than the other. Hence, the DQN can succeed to find shortterm strategies relatively fast but we have to strive to put `gamma` as closest to one as possible to get the best policy. DQN sometimes succeeds but mostly doesn't do better if `gamma` is *too* close to 1 or is set exactly at one; training may become harder as it is trying to really look *far* ahead and approximate a more complicated Q-function. That is why we see models trained with `gamma` around `0.99` even though the argument applied above still applies: Theoretically, the best strategy is the one coming from `gamma=1`.

How can we do better on this issue? One idea is to use the fact that DQN can do well on smaller than 1 discount factor. It may in fact sometimes be better to follow the action provided by a _smaller_ `gamma` like `0.9` than the one provided with `0.99`. The reason is that the action provided by `0.99` may not be the actual best action as the DQN may not have accurately estimated its reward. But it has a higher probability to have accurately approximated the reward for the action from `gamma=0.9`, as it is coming from a more shortterm reward based optimal strategy and therefore easier to approximate. 


Hence, the idea is to apply the Multi-Horizon network detailed in [here](https://arxiv.org/1902.06865) (see p.11, box 3 and 4):

ATTACH BOXES


In our implementation (`DQN.py`), `MHQNetwork` computes the Q-function for multiple values of `gamma`, but instead of choosing a fixed (*hyperbolic*) weighting as done by the authors in that paper, we let the `DFQNetwork` _learn_ the best weighting. The loss function for `DFQNetwork` is given assuming it is trying to find a linear combination which gives optimal strategy for `gamma=1` (which was our primary goal).

The weights of `DFQNetwork` have another nice interpretation: They give a *learned* discount function, call it `Gamma(t)` (dependent on timestep `t`), which is no longer exponential. This can be thought of as a preferred discount function for the DQN with which it *can* be trained and get close to the theoretical optimal policy provided by `gamma=1`. Below is an analysis similar to the one in the paper, which shows why `DFQNetwork` can be interpreted as the learned discount function:

ATTACH THE ANALYSIS HERE


The agent trains in 270 episodes (or 170) with the same hyperparameters as vanilla DQN, and can be trained in 480 episodes to score 16. Further, we can watch either any of the horizons playing or their linear combination. We play some of the low and high `gamma` values and their linear combination. Each of them seem to achieve a similar result. We can plot the `Gamma(t)` function at score 13 and 16.

ATTACH THE PLOTS OF GAMMA HERE

We see quite an uptick where in the first 4 seconds `Gamma(t)` goes higher than 1! This means the agent has become quite certain about the next 4 seconds rewards as the multi horizons have been able to provide the agent an accurate prediction of the total reward for the next 4 seconds. But ultimately the weighting decreases faster and moves asymptotically to almost 80% of the exponential weighting as shown below. This shows the agent has a lot of *mistrust* in the multihorizons predictions on whether it can get rewards on those long timescales.

ATTACH THE GAMMA/EXP HERE

 
##### &nbsp;

### 5. Time Awareness 

Although it may seem that our problem is an instance of completely observable environment, we have one crucial aspect of the environment missing in the `state` given by the `env`: the remaining time of the episode. It is obvious how this could influence the optimal behavior. If the agent sees a yellow banana close ahead and a large number of bananas to its far right, it may choose to collect the bananas to the far right or try to waste some time to look around while it does not have any idea of the remaining time. This is certainly not optimal in such an episodic task. To remedy this, we can add the time remaining to the state. The idea was first proposed in [here](https://arxiv.org/abs/1712.00378). Combining this with the multihorizon could lead to a time aware agent which is trying to maximize the total reward, a combination which is needed as maximizing total reward *without an observation of the time* may not actually be optimal in an episodic task (esp. with a fixed time constraint). Using a `time_step` normalized counter, we can give this information to the agent. We solve the environment in 288 episodes.


ATTACH TRAINING PLOT WITH DESCRIPTION


##### &nbsp;

### 6. Possible Future Improvements 

1. One can imagine an action dependent learnable discount factor. The same analysis applies and we only need to change the `DFQNetwork` to have a separate linear layer for each action. This broadens the scope of strategies the agent can achieve.
2. What would happen if the `DFQNetwork` was deep? It is not clear what meaning of this would be. The analysis in above does not apply (as far as we can see), meaning that nonlinear combination of multihorizon is not associated to a more complicated discount function. Still, it is of theoretical interest to be investigated. Another interpretation of what has been done above and could be done with multilayer `DFQNetwork` is that we are doing a certain kind of *hierarchical RL* by combining multihorizons strategies in a nonlinear way.
3. Vanilla `DQN` seems to be doing a good job, trains in fewer episodes but also trains faster (in real-time) than other models due to its simplicity. But it could be improved using a variety of improvements all implemented in Rainbow DQN such as:
    - [Prioritized Replay](https://arxiv.org/abs/1511.05952): To learn from important experience that may get lost as the memory is finite, we add `|delta_t|` (the absolute value of TD error at timestep `t`) to the `SARS` tuple and order by `|delta_t|` the probability of their selection. To avoid zero division we need to add a decay term `eps` to all `|delta_t|`. We can also move along a spectrum of unifrom-prioritized sampling by introducing a hyperparameter `0 <= a <= 1` and taking the probability to come from `p_t**a=(|delta_t|+eps)**a`, i.e. `P(t)=p_t**a / sum_t p_t**a`. When `a=1`, this is the pure prioritized sampling and when `a=0` it is the usual uniform sampling. Finally, as our distribution is no longer iid, we need to change the learning rule and multiply the learning rate by `1/N * 1/P(t)` where `N` is the replay buffer size.
    - [Dueling DQN](http://proceedings.mlr.press/v48/wangf16.pdf): We know that `Q(s,a)=V(s)+A(s,a)` where `V` is the state value function and `A` is the advantage function. The *state* value function by definition do not differ across actions. Therefore the idea is to estimate them as they are easier to estimate and then try to approximate the difference across actions which is the advantage function. The implementation is best summarized in the following picture:   ATTACH PICTURE HERE  More details can be found in the paper. For example, we want to make sure that our estimation of `V,A` is such that `max_a Q(s,a) = V`. To ensure this *indentifiability* of `V`, it is better to use `A - max_a A` instead of just `A` in the last addition module above. An even better solution is to use `A- mean(A)` as it would force the advantage to only move as fast as the mean instead of compensating for maximum. When acting, we would only need to evaluate the max in the advantage stream as it is the only of the two streams dependent on action and we are confident enough that always `max_a Q(s,a) = V`. Further, it should be noted that in Dueling DQN, the Q-value is essentially updated for all actions as they share information in `V` and `mean(A)`, instead of just one action like in DQN. As the `action_size` grows, the outperformance of Dueling DQN also grows, as the advantage stream makse sure that noise cannot abruptly change the policy when scale of gap between `Q(s,a)`s for all `a`, is much smaller than the scale of `Q(s,a)` itself. 
    - [Double DQN](https://arxiv.org/abs/1509.06461): When it comes to picking the target, DQN picks the max of the target network Q-values. This could result in overestimation as we have not yet gathered enough information (esp. at the beginning of training). So instead, we could choose the best action according to our local DQN, and then evaluate the target DQN at that action. Hence we are using a different set of parameters (the local DQN) to select the action from the parameters used to evaluate it (the target DQN). This results in stability and avoids the explosion of Q-values. A small change in the code where we select `next_action_values` in `agent` can be performed to apply this algorithm.


4. Using the above approaches, we could try to beat the record of vanilla `DQN` in solving the environment in as few episodes as possible, but also try to reach the highest possible score; we only went up to score 16 as the agents were beginning to show signs of destabilization.



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

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  
