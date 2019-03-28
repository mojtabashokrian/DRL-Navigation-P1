import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque 
from hyperparameters import *


def dqn_train(agent,env,brain_name,n_episodes=2000, max_t=1000, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
        target_score=13,mh_take_action=False,which_mh=None):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode (for Banana environment, an episode is 300 timesteps)
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        target_score: stopping point for training and saving checkpoint
        mh_take_action,which_mh: parameters to be passed to agent.act, see Agent.py 
    Returns
    ======
        scores,avgs: The history of scores and the history of last 100 score averages
        Also saves the checkpoint and the scores array (filepath `stats` given by agent (hyper)parameters)
    """
    stats=f"{agent.buffer_size}{agent.batch_size}{agent.gamma}{agent.tau}{agent.lr}{agent.update_every}"
    stats+=f"{agent.hidden_layer}{agent.hidden_layer_size}{agent.seed}{agent.network}{agent.time_aware}"
    if agent.network=="mhq_dfq": stats+=f"{agent.mh_size}{agent.hidden_layer_d}"
    stats+=f"{eps_start}{eps_end}{eps_decay}"  #save_path a la tensorboard! 
    scores=[]# list containing scores from each episode
    avgs=[]# list containing average of last 100 scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps, mh_take_action, which_mh,t)
            env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]  
            agent.step(state, action, reward, next_state, done,t)
            state = next_state
            score += reward
            if done:
                break       
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
        if np.mean(scores_window)>=target_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if agent.network=="mhq_dfq":
                torch.save(agent.mhqnetwork_local.state_dict(), f"{stats+str(i_episode)}mhq_local.ph")
                torch.save(agent.dfqnetwork_local.state_dict(), f"{stats+str(i_episode)}dfq_local.ph")
                torch.save(agent.mhqnetwork_target.state_dict(), f"{stats+str(i_episode)}mhq_target.ph")
                torch.save(agent.dfqnetwork_target.state_dict(), f"{stats+str(i_episode)}dfq_target.ph")
            else:
                torch.save(agent.qnetwork_local.state_dict(), f"{stats+str(i_episode)}q_local.ph")
                torch.save(agent.qnetwork_target.state_dict(), f"{stats+str(i_episode)}q_target.ph")
            np.savetxt(f'{stats+str(i_episode)}scores',np.array(scores))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            avgs=np.array([np.array(scores[max(i-100,0):i]).mean() for i in range(1,len(scores)+1)])
            plt.plot(np.arange(len(avgs)), avgs, c='r')
            plt.ylabel('scores')
            plt.xlabel('Episode #')
            plt.show()
            break
    return scores,avgs