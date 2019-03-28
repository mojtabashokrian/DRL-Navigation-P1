import numpy as np
import matplotlib.pyplot as plt

def AgentPlay(agent,env,brain_name,train_mode=False,counter_states=3,view_episodes=1,
              loop_breaker=False,mh_take_action=False,which_mh=None):
    """
    Watch the agent play in Unity: 
            train_mode: if you want to watch the agent play, set False, if you want to run many plays in short time, set True
            counter_states: How many times a state is observed before a loop is counted
            loop_breaker: implement the loop breaking mechanism if a state is observed more than counter_states
            Returns score_list, loop_list
            mh_take_action, which_mh: passed to agent.act (besides state and eps) 
    """
    score_list = []
    loop_list=[]
    for i_episode in range(1, view_episodes+1): 
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
          # initialize the score
        score_list.append(0)
        states_list=[]
        loop_list.append(0)
        states_list.append(state)
        eps=0.
        time_step=0
        if i_episode%10==0:
            print('epsiode {}, number of loops so far {}, average score so far {}'.format(i_episode,np.sum(np.array(loop_list)),np.average(np.array(score_list))))
        while True:
            # select an action
            if agent.time_aware==True:
                action = agent.act(state,eps,mh_take_action,which_mh,time_step=time_step) 
            else:
                action = agent.act(state,eps,mh_take_action,which_mh)
            env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score_list[i_episode-1] += reward                                # update the score
            state = next_state
            if np.sum(np.abs(np.sum(np.array(states_list)-state,1)).squeeze()<1e-5)>counter_states:
                loop_list[i_episode-1]+=1
                if loop_breaker==True: #implementing loop_breaker
                    eps=0.99
            states_list.append(state)
            time_step+=1
            if done:                                       # exit loop if episode finished
                break

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(score_list)), score_list)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(loop_list)), loop_list)
    plt.ylabel('Loop')
    plt.xlabel('Episode #')
    plt.show()
    return score_list,loop_list