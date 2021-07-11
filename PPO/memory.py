import torch

class Memory:
    def __init__(self, env, agent_policy, opt) -> None:
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        self.states = torch.zeros((opt.update_timestep*opt.num_agents, num_inputs)).to(opt.device).share_memory_()
        self.actions = torch.zeros(opt.update_timestep*opt.num_agents).to(opt.device).share_memory_()
        self.logprobs = torch.zeros(opt.update_timestep*opt.num_agents).to(opt.device).share_memory_()
        self.disreturn = torch.zeros(opt.update_timestep*opt.num_agents).to(opt.device).share_memory_()

        self.agent_policy = agent_policy


        
