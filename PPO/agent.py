import torch.multiprocessing as mp
import torch
from collections import namedtuple
import os
import gym
import time

from torch.nn.modules import module
from .PPO import PPO

MsgRewardInfo = namedtuple('MsgRewardInfo', ['agent', 'episode', 'reward'])
# for when agent reached update timestep
MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'update'])
# for when agent reached max episodes
MsgMaxReached = namedtuple('MsgMaxReached', ['agent', 'reached'])


class Agent(mp.Process):
    def __init__(self, name, envname, memory, pipe, opt) -> None:
        mp.Process.__init__(self, name=name)

        self.proc_id = name
        self.memory = memory
        self.pipe = pipe

        self.max_episode = opt.max_episode
        self.max_timestep = opt.max_timestep
        self.update_timestep = opt.update_timestep
        self.log_interval = opt.log_interval

        self.gamma = opt.gamma
        self.env = gym.make(envname)

    def run(self):
        print("Agent {} start, Process id {}", self.name, os.getpid())
        actions = []
        rewards = []
        states = []
        logprobs = []
        dones = []
        timestep = 0

        running_reward = 0

        for i_episodes in range(self.max_episode):
            state = self.env.reset()

            for i in range(self.max_timestep):
                timestep += 1
                states.append(state)

                with torch.no_grad():
                    action, logprob = self.memory.agent_policy.act(state, False)
                state, reward, done, _ = self.env.step(action)

                actions.append(action)
                logprobs.append(logprob)
                rewards.append(reward)
                dones.append(done)

                running_reward += reward

                if timestep % self.update_timestep == 0:
                    stateT, actionT, logprobT, disreturn = self.experience_to_tensor(
                        states, actions, rewards, logprobs, dones
                    )
                    self.add_experience_to_pool(stateT, actionT, logprobT, disreturn)
                    msg = MsgUpdateRequest(int(self.proc_id), True)
                    self.pipe.send(msg)
                    msg = self.pipe.recv()

                    timestep = 0
                    actions = []
                    rewards = []
                    states = []
                    logprobs = []
                    dones = []
                if done:
                    break

                time.sleep(0.005)
                self.env.render()
            if i_episodes % self.log_interval == 0:
                running_reward = running_reward/self.log_interval
                msg = MsgRewardInfo(self.proc_id, i_episodes, running_reward)
                self.pipe.send(msg)
                running_reward = 0
                    
        msg = MsgMaxReached(self.proc_id, True)
        self.pipe.send(msg)
        print("Agent {} over, Process id {}", self.name, os.getpid())

    def experience_to_tensor(self, states, actions, rewards, logprobs, dones):
        stateTensor = torch.tensor(states).float()
        actionTensor = torch.tensor(actions).float()
        logprobTensor = torch.tensor(logprobs).float()

        discounted_reward = 0
        disReturnTensor = []

        for reward, done in zip(reversed(rewards),reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma*discounted_reward)
            disReturnTensor.insert(0, discounted_reward)

        disReturnTensor = torch.tensor(disReturnTensor).float()
        return stateTensor, actionTensor, logprobTensor, disReturnTensor

    def add_experience_to_pool(self, stateTensor, actionTensor, logprobTensor, disReturnTensor):
        start_idx = int(self.name)*self.update_timestep
        end_idx = start_idx + self.update_timestep
        self.memory.states[start_idx:end_idx] = stateTensor
        self.memory.actions[start_idx:end_idx] = actionTensor
        self.memory.logprobs[start_idx:end_idx] = logprobTensor
        self.memory.disreturn[start_idx:end_idx] = disReturnTensor
        