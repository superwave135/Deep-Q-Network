import gym

env = gym.make('CartPole-v1')
[next_state, reward, done, info] = env.reset()
print([next_state, reward, done, info])

done=False
  
while(not done):
      action = env.action_space.sample()
      next_state, reward, done, info=env.step(action)
      print(next_state, reward, done, info)
      env.render()

env.close()