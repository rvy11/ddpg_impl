import copy
import math
import os
from collections import namedtuple
import argparse
import json

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ReplayMemory import ReplayMemory
from NNDefs import Critic, Actor
from ActionNoise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import robel

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 4]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

envs = {
    'ant': gym.make('Ant-v2'),
    'dkitty': gym.make('DKittyWalkFixed-v0')
}

default_params = {
    "policy":"DDPG",
    "env":"dkitty",
    "seed":0,
    "start_timesteps":1e4,
    "eval_freq":5e3,
    "max_timesteps":1500000,
    "expl_noise":0.1,
    "replay_size":1000,
    "batch_size":256,
    "tau":0.005,
    "policy_noise":2,
    "action_noise":"Normal",
    "min_action":-1.0,
    "max_action":1.0
}

# Draw the environment while running simulation using actions
# taken from the provided policy or at random if not provided
def render(env, policy=None):
    if policy is None:
        def policy(state):
            return env.action_space.sample()

    state = env.reset()
    env.render()

    while True:
        action = policy(torch.FloatTensor(state)).cpu().data.numpy()
        print(action)
        state, _, done, _ = env.step(action)
        env.render()

        if done:
            print("hit done")
            break

    env.close()

# Return a smoothed dataset by calculating the rolling average
# with the given window size
def rolling_average(data, window_size):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

# Train actor and critic models from batch of experience taken fron replay buffer
def train_ddpg_batch(critic_optimizer, actor_optimizer, batch, critic_model, critic_target, actor_model, actor_target, gamma, tau):
    current_Q = critic_model(batch.states, batch.actions)
    is_non_terminal = (batch.dones == False)
    target_Q = critic_target(batch.next_states, actor_target(batch.next_states.to(device)))
    target_Q = batch.rewards + (gamma*target_Q*is_non_terminal.float()).detach()

    # computing the scalar MSE loss between computed values and the TD-target
    critic_loss = F.mse_loss(current_Q, target_Q)

    # computing actor loss (negative value to do SGD)
    actor_loss = -critic_model(batch.states, actor_model(batch.states)).mean()
    
    # Critic Update
    critic_optimizer.zero_grad()  # reset all previous gradients
    critic_loss.backward()  # compute new gradients
    critic_optimizer.step()  # perform one gradient descent step
    
    # Actor Update
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Update the frozen target models
    for param, target_param in zip(critic_model.parameters(), critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for param, target_param in zip(actor_model.parameters(), actor_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    return critic_loss.item()

# Main function for DDPG training
def train_ddpg(
    env,
    num_steps,
    replay_size,
    batch_size,
    gamma,
    noise,
    num_saves=5,
    replay_prepopulate_steps=0,
    lr_critic=1e-2,
    lr_actor=1e-4, 
    tau=0.001,
    max_action=1.0,
    min_action=-1.0
):
    # get the state_size from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # initialize the Critic and target Critic models
    critic_model = Critic(state_size, action_size).to(device)
    critic_target = copy.deepcopy(critic_model)

    # initialize the Actor and target Actor models
    actor_model = Actor(state_size, action_size, 1.0).to(device)
    actor_target = copy.deepcopy(actor_model)

    # initialize the optimizer
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), weight_decay=lr_critic)
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=lr_actor)

    # initialize the replay memory and prepopulate it
    memory = ReplayMemory(replay_size, state_size, action_size)
    memory.populate(env, replay_prepopulate_steps)

    # initiate lists to store returns, lengths and losses
    returns = []
    lengths = []
    losses = []

    # initiate structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}
    saved_policies = {}

    i_episode = 0  # use this to indicate the index of the current episode
    t_episode = 0  # use this to indicate the time-step inside current episode
    G = 0 # initializing return variable (incremental tally during each episode)

    state = env.reset()  # initialize state of first episode

    # iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps)
    for t_total in pbar:
        # use t_total to indicate the time-step from the beginning of training
        # save model
        if t_total in t_saves:
            model_name = '%04.1f' % (100 * t_total / num_steps)
            model_name = model_name.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(critic_model)
            saved_policies[model_name] = copy.deepcopy(actor_model)

        action_arr = actor_model(torch.FloatTensor(state).to(device)).cpu().data.numpy()

        action = np.clip(action_arr+noise(), a_min=min_action, a_max=max_action)

        ss, r, done, info = env.step(action)
        memory.add(state=state, action=action, reward=r, next_state=ss, done=done)
        
        batch = memory.sample(batch_size)
        loss = train_ddpg_batch(critic_optimizer, actor_optimizer, batch, critic_model, critic_target, actor_model, actor_target, gamma, tau)
        losses.append(loss)

        if done:
            # When episode is done, collect return, update counters, and reset env
            G += (gamma**t_episode)*r
            returns.append(G)
            
            pbar.set_description(
                'Episode: %d | Steps: %d | Return: %5.2f' % (i_episode, t_episode + 1, G)
            )
            
            lengths.append(t_episode+1)
            t_episode = 0
            i_episode += 1
            G = 0
            state = env.reset()
            
        else:
            # While episode is not done, move state pointer forward and update return
            state = ss
            G += (gamma**t_episode)*r
            t_episode += 1
                                          
    saved_models['100_0'] = copy.deepcopy(critic_model)
    saved_policies['100_0'] = copy.deepcopy(actor_model)

    return (
        saved_models,
        saved_policies,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', help='Run simulation using provided trained model')
    parser.add_argument('-f', '--configfile', help='Use provided config (JSON) file for training')
    args = parser.parse_args()

    # Post-Training Simulation Mode
    if args.run != None:
        print("Running simulation using trained model "+args.run)
        env_name = [value for value in list(envs) if value in args.run.split('_')]
        if len(env_name) != 0:
            env = envs[env_name[0]]
            try:
                checkpoint = torch.load(args.run, map_location=torch.device('cuda'))
                policy = Actor.custom_load(checkpoint['100_0'])
                render(env, policy)
            except FileNotFoundError:
                pass
        else:
            print('Can\'t open environment with the given trained model')
    # Training mode
    else:
        params = default_params

        if args.configfile != None:
            configfile = args.configfile
        else:
            configfile = 'config.json'
        
        if os.path.exists(configfile):
            with open(configfile) as config_file:
                config_params = json.load(config_file)
                for key in config_params.keys():
                    params[key] = config_params[key]

        print(params)
        
        env = envs[params['env']]
        gamma = 0.99
        num_steps = params['max_timesteps']
        num_saves = 5  # save models at 0%, 25%, 50%, 75% and 100% of training
        replay_size = params['replay_size']
        replay_prepopulate_steps = params['start_timesteps']
        batch_size = params['batch_size']
        min_action = params['min_action']
        max_action = params['max_action']

        # get the state and action sizes from the environment
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        if params['action_noise'] == "OU":
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.ones(action_size)*((max_action+min_action)/2.0), sigma=0.2, theta=0.15, dt=0.1)
        else:
            action_noise = NormalActionNoise(mu=np.ones(action_size)*((max_action+min_action)/2.0), sigma=max_action*params['expl_noise'])

        # Run DDPG
        saved_models, saved_policies, returns, lengths, losses = train_ddpg(
            env=env,
            num_steps=num_steps,
            num_saves=num_saves,
            replay_size=replay_size,
            replay_prepopulate_steps=replay_prepopulate_steps,
            batch_size=batch_size,
            gamma=gamma,
            max_action=params['max_action'],
            min_action=params['min_action'],
            noise=action_noise
        )

        assert len(saved_models) == num_saves
        assert all(isinstance(value, Critic) for value in saved_models.values())
        assert len(saved_policies) == num_saves
        assert all(isinstance(value, Actor) for value in saved_policies.values())

        # Save actor and critic models
        checkpoint_critic = {key: critic.custom_dump() for key, critic in saved_models.items()}
        torch.save(checkpoint_critic, 'saved_models/checkpoint_critic_%s_%s_noise_%s.pt' % (params['env'], params['policy'], params['action_noise']))
        checkpoint_actor = {key: actor.custom_dump() for key, actor in saved_policies.items()}
        torch.save(checkpoint_actor, 'saved_models/checkpoint_actor_%s_%s_noise_%s.pt' % (params['env'], params['policy'], params['action_noise']))

        # Save plots of performance metrics
        plt.rcParams['figure.figsize'] = [20, 6]
        num_episodes = len(returns)
        num_loss_calc = len(losses)
        plt.subplot(1, 3, 1)
        plt.plot(range(num_episodes), returns, 'r', label='Raw', alpha=0.4)
        plt.plot(range(num_episodes), rolling_average(data=returns, window_size=100), 'b', label='Smoothed')
        plt.legend()
        plt.ylabel('Return')
        plt.xlabel('Number of Episodes')
        plt.title('%s %s Performance' % (params['env'], params['policy']))
        plt.subplot(1, 3, 2)
        plt.plot(range(num_episodes), lengths, 'r', label='Raw', alpha=0.4)
        plt.plot(range(num_episodes), rolling_average(data=lengths, window_size=100), 'b', label='Smoothed')
        plt.legend()
        plt.ylabel('Length of Episode')
        plt.xlabel('Number of Episodes')
        plt.title('%s %s Episode Lengths' % (params['env'], params['policy']))
        plt.subplot(1, 3, 3)
        plt.plot(range(num_loss_calc), losses, 'r', label='Raw', alpha=0.4)
        plt.plot(range(num_loss_calc), rolling_average(data=losses, window_size=100), 'b', label='Smoothed')
        plt.legend()
        plt.ylabel('MSE Losses')
        plt.xlabel('Number of Steps')
        plt.title('%s %s Episode MSE Losses' % (params['env'], params['policy']))
        plt.savefig('plots/%s_%s_noise_%s_performance.png' % (params['env'], params['policy'], params['action_noise']))