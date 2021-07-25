# import packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Deep Model Used by the Agent
class Model:
    def __init__(self, state_size, action_size):
	    # state_size -- input, action_size -- output
        self.state_size = state_size
        self.action_size = action_size

		  # Build a neural network model and assign it to the variable self.model
		  # e.g.  self.model = Sequential()
		  # cause the size of state is just 4, a convnet seems not a good choice
		  # fully connected multi-layered feed forward neural network can be considered
        
        # # # Your code goes here
        self.model = Sequential()
        
        # ‘Dense’ is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 24 nodes
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        
        # Hidden layer with 24 nodes
        self.model.add(Dense(24, activation='relu'))
        
        # Output Layer with # of actions: 2 nodes (left, right)
        self.model.add(Dense(self.action_size, activation='linear'))
        
        # Create the model based on the information above
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
			
		
        
    def _save_model(self):
        self.model.save_weights("weights_cartpoledqn.h5")
        
    def _load_model(self):
        self.model.load_weights("weights_cartpoledqn.h5")
 


