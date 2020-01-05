import gym
import spinup
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

def run_game(model,env):
    # game loop
    # get experience
    obs = env.reset()
    done = False
    experience = []
    while(not done):
        action,value = model.predict(obs)
        action = np.argmax(action.detach().numpy())
        env.render()
        new_obs,reward,done,info = env.step(action)

        experience.append([obs,action,new_obs,reward])

        obs = new_obs

    env.close()

    return experience


class pytorch_ppo_module(torch.nn.Module):
    def __init__(self,input_size,number_of_actions):
        super(pytorch_ppo_module,self).__init__()
        self.input_layer = nn.Linear(input_size,256)
        self.hidden1 = nn.Linear(256,256)
        self.hidden2 = nn.Linear(256,512)
        self.hidden3 = nn.Linear(512,512)
        self.action_output = nn.Linear(512,number_of_actions)
        self.value_output = nn.Linear(512,1)

    def forward(self,obs):
        x = self.input_layer(obs)
        x = self.hidden1(x)
        x = self.hidden2(x)
        hidden = self.hidden3(x)

        action_output = self.action_output(hidden)
        value_output = self.value_output(hidden)

        return action_output,value_output

        
    
    
class ppo_wrapper:
    def __init__(self,input_size,number_of_actions,std=1.0):
        # initialize neural network to produce gausians 
        self.input_size = input_size
        self.number_of_actions = number_of_actions
        self.module = pytorch_ppo_module(input_size,number_of_actions)
        self.old_module = self.copy_current_module()
        self.epsilon = .2

        self.optimizer = Adam(self.module.parameters(),lr=.001)
        self.value_loss = nn.MSELoss()

        self.std = torch.ones((1,number_of_actions))*std

    def predict(self,obs):
        obs = torch.Tensor(obs)
        action_output,value_output = self.module(obs)
        mu = action_output #mean of the value of the continous output
        std = self.std.exp()

        return action_output,value_output

    def predict_old_policy(self,obs):
        obs = torch.Tensor(obs)
        action_output,value_output = self.module(obs)
        return action_output,value_output



    def copy_current_module(self):
        result_model = pytorch_ppo_module(self.input_size,self.number_of_actions)
        result_model.load_state_dict(self.module.state_dict())
        return result_model

    def train(self,experience):
        # Loss
        # rt(policy_params) = new_policy(action|state)/ old_policy(action|state)
        # rt(old_policy_params) = 1
        # At = advantage function = Q(s,a) - V(s)
        # L_clip = E[min(rt(theta)*At, clip(rt(theta), 1-e,1+e)*At
         
        # At is how much better than the general state our action is
        L_clip_loss = 0
        state_values  = []

        obs,action,new_obs,reward  = zip(*experience)
        # Advantage =  Q(s,a) - V(s)
        # Q(s,a) = V(s_t+1)
        next_state_value = self.predict(new_obs)[1]
        this_state_value = self.predict(obs)[1]
        advantage = next_state_value - this_state_value


            # rt(policy_params) = new_policy(action|state) / old_policy(action|state)
        rt = torch.exp(self.predict(obs)[0]-self.predict_old_policy(obs)[0].detach())

        L_clip_loss = -torch.min(rt*advantage,torch.clamp(rt,1-self.epsilon,1+self.epsilon)*advantage)



        #L_clip_loss = -L_clip_loss 

        obs,action,new_obs,rewards = zip(*experience)
        #print(self.predict(obs)[1])
        value_loss = self.value_loss(self.predict(obs)[1],torch.flatten(torch.Tensor(rewards)))
        total_loss = (L_clip_loss + value_loss ).mean()



        # pytorch update step
        self.optimizer.zero_grad()

        # loss is now mse
        total_loss.backward()
        print(total_loss)

        self.optimizer.step()

        self.old_module = self.copy_current_module()

            
def fix_reward(experience):
    reward_after_x = 0
    for i in range(len(experience)-1,-1,-1):
        reward_after_x += experience[i][3]
        experience[i][3] = reward_after_x
    return experience





if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    print()
    print(env.action_space)
    print()

    obs = env.reset
    model = ppo_wrapper(4,2)
    print(model)

    while True:
        experience = run_game(model,env)
        #experience = fix_reward(experience)
        model.train(experience)

