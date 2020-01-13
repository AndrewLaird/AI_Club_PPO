import gym
import numpy as np 
import collections 
import torch 
from torch import nn
from torch import dist
from torch import optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

def run_game(env,model=False,render=False,max_frames=100000):
    state = env.reset()
    frames = 0
    entropy = 0
    done = False

    states    = []
    actions   = []
    rewards   = []
    log_probs = []
    values    = []
    masks     = []

    experience = []
    while(not done and frames < max_frames):

        state = torch.FloatTensor(state)#.to(device)

        dist, value = model(state)
        action = dist.sample()[0]

        new_state,reward,done,info = env.step(action)

        if(render):
            env.render()

        frames += 1


        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()


        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        masks.append((1 - int(done)))#.to(device))


        state = new_state

    return states,actions,rewards,log_probs,values,masks



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=1.0):
        # Pytorch Setup of a module
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x).reshape(1,-1)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    # From: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
    
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    # mini_batch_size=batch_size just means we shuffle the data
    mini_batch_size = batch_size
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        output_states = states[rand_ids]
        output_actions = actions[rand_ids]
        output_log_probs = log_probs[rand_ids]
        output_returns = returns[rand_ids]
        output_advantage = advantage[rand_ids]
        yield output_states, output_actions, output_log_probs, output_returns, output_advantage



def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for epoch in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            print("Loss:",loss)
            # Suspect retain_graph
            loss.backward()
            optimizer.step()
            """
    print("done training")
    return model

def plot(frame_idx, rewards):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def test_env(env,model,vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def run_trainer(env,model,optimizer):
    # Constants
    # max_frames to train on 
    max_frames = 1000000
    frame_idx  = 0
    test_rewards = []
    # number of steps per game
    num_steps = 100
    #Hyper params:
    num_steps        = 100
    mini_batch_size  = 200
    ppo_epochs       = 100
    threshold_reward = 190

    state = env.reset()
    early_stop = False
    counter = 0

    while not early_stop:
        counter += 1

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        returns   = []
        entropy = 0
        obs = env.reset()

        for step in range(num_steps):
            print("game:",step+1)
            game_states,game_actions,game_rewards,game_log_probs,game_values,game_masks = run_game(env,model=model)
            frame_idx += len(game_states)

            # Compute the returns
            # used for advantage
            next_state = torch.FloatTensor(game_states[-1])#.to(device)
            _, next_value = model(next_state)
            next_value = next_value.detach()
            game_returns = compute_gae(next_value, game_rewards, game_masks, game_values)


            # update all our information updated
            states.extend(game_states)
            actions.extend(game_actions)
            log_probs.extend(game_log_probs)
            rewards.extend(game_rewards)
            # will turn into advantage
            values.extend(game_values)
            masks.extend(game_masks)
            returns.extend(game_returns)




        states_tensor    = torch.stack(states,axis=0)
        actions_tensor   = torch.stack(actions)
        returns_tensor   = torch.cat(returns).view(-1).detach()
        log_probs_tensor = torch.cat(log_probs).detach()
        values_tensor    = torch.cat(values).detach()
        advantage_tensor = returns_tensor - values_tensor

        print("ppo update")
        

        model = ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantage_tensor)

        test_reward = np.mean([test_env(env,model) for _ in range(5)])
        print("reward",test_reward)
        test_rewards.append(test_reward)
        #plot(frame_idx, test_rewards)
        run_game(env,model=model,render=counter%1==1)
        if test_reward > threshold_reward: early_stop = True
        #test_env(env,model,vis=True)

        print("updating the model")
        print("---------")
        # save the model
        torch.save(model,"best_model.torch")
        

if (__name__ == "__main__"):
    #env = gym.make("MountainCarContinuous-v0")
    #env = gym.make("BipedalWalker-v2")
    #env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("Pendulum-v0")


    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    hidden_size = 512
    lr =1e-3#3e-4


    model = ActorCritic(num_inputs, num_outputs, hidden_size,std=0.0)#.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    run_trainer(env,model,optimizer)



    # play 5 games to get started
