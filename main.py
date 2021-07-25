import gym
import agent
from agent import Agent
import numpy as np
from time import perf_counter   

score_list = []	
############  start the episodes, fill the memory pool and train	
def train():

    # initialize gym environment
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    cartpole_agent = agent.Agent(state_size, action_size)
    
    # Parameters
    batch_size = 32  # size of the batch sampled from memory pool, for updating the weights
    maxscore = 0     # for saving the weights with highest score 
    EPISODES = 200   # play the game 200 times

    # Iterate the game
    for e in range(EPISODES):
        t1_start = perf_counter() # Start the stopwatch / counter
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        done = False

        # t : steps, the maximal score is 200, terminate the episode if done
        for t in range(200):
            
			# Decide action: epsilon greedy
            action = cartpole_agent.eps_greedy(state)
			
			# take action, get the next state and reward              
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            # append the transition(previous state, action, reward, state, done) in the memory pool 
            cartpole_agent.mempool.memorize(state, action, reward, next_state, done)

            # make next_state the new current state
            state = next_state
            
            # time to turn to the next episode
			      # but before the next episode, to print out info of current episode and save the weights of the best model			
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}".format(e, EPISODES, t))
                
                if maxscore < t:
                    maxscore = t
                    cartpole_agent.net._save_model()
                    score_list.append(maxscore)
                break
            
		        # start train only if samples in memory pool are sufficient for learning 
            if cartpole_agent.mempool.poollen() > batch_size:
                cartpole_agent.train_policy_net(batch_size)  
                  
        t1_stop = perf_counter() # Stop the stopwatch / counter 
        print(f'  Elapsed time taken {t1_stop - t1_start} seconds\n')

    env.close()
    
def play():

    # initialize gym environment
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    cartpole_agent = agent.Agent(state_size, action_size)
	
	  # Load weights for the model
    cartpole_agent.net._load_model()
    
    observe = env.reset()
    state = np.reshape(observe, [1, 4])
    done = False
    score = 0
  
    while(not done):
      action = cartpole_agent.get_action(state)
      observe, reward, done, _ = env.step(action)
      state = np.reshape(observe, [1, 4])
      env.render()
      score+=1

    env.close() 
    print("score:", score)
    
train()
# play()

# import matplotlib.pyplot as plt
# # Plot for Scores vs num of Episodes
# plt.figure(figsize=(12,8))
# plt.plot(score_list)
# plt.title('Plot of Scores over number of Episodes')
# plt.ylabel('Score')
# plt.xlabel('Number of Episodes')
# # plt.legend(loc='lower center', fontsize = 'x-large')
# plt.savefig("weights_cartpoledqn_plot.png", bbox_inches="tight")
# plt.show()