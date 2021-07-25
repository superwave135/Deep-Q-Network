#import packages
import model
from model import Model
import memory
from memory import Memory
import numpy as np
import random

# Deep Q-learning Agent for cartpole

class Agent:

    def __init__(self, state_size, action_size):    
        self.state_size = state_size
        self.action_size = action_size

        # create memory pool
        self.mempool = memory.Memory()
        
        # create neural network model, we call it policy net
        self.net = model.Model(state_size=state_size, action_size=action_size)
        
        # parameters for q-learning
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01       # the low bound of espison
        self.epsilon_decay = 0.995    # used for adjusting the epsion along with the time/steps
                
    def eps_greedy(self, state):

        # get action using epsilon greedy policy together with policy net
        # self.epsilon can be used for the greedy policy
        # return the action
        # # # Your code goes here
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.net.model.predict(state)
            return np.argmax(act_values[0])

        
    def get_action(self, state):
        
        # get action just using policy net
        # return the action
        # # # Your code goes here
        act_values = self.net.model.predict(state)
        return np.argmax(act_values[0])

            
    def train_policy_net(self, batch_size):

        # Get minibatch from memory pool 
        minibatch = self.mempool.sample_minibatch(batch_size)

        # initialize input and output
        # states, targets = [], []
        #--------------------------------------------------------------------------------
        ## generate training data set for training the policy net
        # for state, action, reward, next_state, done in minibatch:
            
        #     # compute the targets, for terminal(done) and non-terminal(not done)
        #     # # # Your code goes here
        #     target = reward
        #     if not done:
        #         target = reward + self.gamma * np.amax(self.net.model.predict(next_state)[0])

        #     # append the state and target to states(inputs) and targets(outputs)
        #     # # # Your code goes here
        #     targetp = self.net.model.predict(state)
        #     targetp[0][action] = target
            
        #     states.append(state[0])
        #     targets.append(targetp[0])
        #-------------------------- vectorized version below ------------------------------------------------------

        # vectorized = True
        # if vectorized:
        #     states = np.array([b[0][0] for b in minibatch])
        #     actions = np.array([b[1] for b in minibatch])
        #     targets = np.array([b[2] for b in minibatch])
        #     next_states = np.array([b[3][0] for b in minibatch])
        #     dones = np.array([b[4] for b in minibatch])
        #     future_rewards = np.array(self.gamma * np.max(self.net.model.predict(np.array(next_states)), axis=1)) # note preds are Nx2

        #     targets = [t if d else (t + f) for t, d, f in zip(targets, dones, future_rewards)]
        #     target_preds = self.net.model.predict(np.array(states)) # Nx2

        #     # print(target_preds.shape, len(actions), len(targets)) # (32,1,2), 32, 32
        #     for i, (t, a) in enumerate(zip(targets, actions)):
        #         target_preds[i][a] = t

        #     targets = target_preds

        #--------------------------- vectorized version above -----------------------------------------------------

        #-------------------------- tianliang version below ------------------------------------------------------
        # generate training data set for training the policy net
        minibatch = np.asarray(minibatch)

        states = np.concatenate(np.concatenate(minibatch[:, 0])).reshape(batch_size, -1)

        actions = minibatch[:,1].astype(int) # shape is (batch_size, )

        rewards = minibatch[:, 2].astype(int) # shape is (batch_size, )

        next_states = np.concatenate(np.concatenate(minibatch[:, 3])).reshape(batch_size, -1)

        dones = minibatch[:, 4].astype(bool) # shape is (batch_size, )

        future_rewards = self.gamma * np.max(self.net.model.predict(next_states), axis=1)

        no_future_rewards = np.zeros(batch_size) # prepare value if done == True

        pred_rewards = np.where(dones, no_future_rewards, future_rewards) + rewards
        pred_rewards = np.vstack((pred_rewards, pred_rewards)).T

        targets = self.net.model.predict(states)

        # every row, the action position value is one, another postion is zero
        action_filter = np.vstack((np.ones(batch_size)-actions, actions)).T

        # two parts: 1st, keep not choosed action pred value, 2nd, change the choosed action value as future rewards.
        targets = targets * (np.ones((batch_size, self.action_size)) - action_filter) + pred_rewards * action_filter

        #--------------------------- tianliang version above -----------------------------------------------------
        # train the policy net
        # # # Your code goes here
        self.net.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)    
        
        # Decay the value of epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
     

  