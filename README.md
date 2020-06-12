# Proximal Policy Optimization (PPO)
This is a project to show off the Reinforcement Learning algorithm PPO. PPO is the sucessor to TRPO (Trust Region Policy Optimization), the intuition behind both of these algorithms is that when we play alot of games we get a better idea of the direction that we should move our policy in. However, because of the nature of reinforcement learning, small changes in our policy could have huge changes in our performace. So to acount of that, both PPO and TRPO try to reign in how far away the new policy is from the last policy  
  
To make a tutorial I applied PPO to the bipedalWalker environment in Open AI's gym. My goal was to show off how fast PPO optimizes a complex environment, this tutorial ran completely during a 40 minute presentation. I was able to show off the trained model that I started at the beginning of the presentation.

- Video link to presentation: https://www.youtube.com/watch?v=YjAKEGXYf_s&t=4s

gif of the final product:
![bipedal2](BipedalWalker2.gif)
