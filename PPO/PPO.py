import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .model import Actor, Critic

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )
        self.set_init()
    
    def set_init(self):
        i = 0
        for layer in self.action_layer:
            i += 1
            if layer.__module__ == "torch.nn.modules.linear" and i == 5:
                # nn.init.normal_(layer.weight,mean=0.,std=0.1)
                layer.weight.data.mul_(0.1)
                layer.bias.data.mul_(0.0)
        for layer in self.value_layer:
            i += 1
            if layer.__module__ == "torch.nn.modules.linear" and i == 11:
                layer.weight.data.mul_(0.1)
                layer.bias.data.mul_(0.0)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, evaluate):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        if evaluate:
            _, action = action_probs.max(0)
        else:
            action = dist.sample()
            # action = action.clamp(-0.4,0.4)
        
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)

        return action_log_probs, torch.squeeze(state_value), dist_entropy


class PPO(nn.Module):
    def __init__(self, env, opt):
        super().__init__()
        self.lr = opt.lr
        self.betas = opt.betas
        self.gamma = opt.gamma
        self.eps_clip = opt.eps_clip
        self.k_epoches = opt.k_epoches

        self.env = env
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.policy = ActorCritic(
            state_dim = num_inputs,
            action_dim = num_actions,
            n_latent_var = opt.n_latent_var
        ).to(opt.device)

        self.actor_optimizer = torch.optim.Adam(
            self.policy.action_layer.parameters(),
            lr = self.lr,
            betas= self.betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.policy.value_layer.parameters(),
            lr = self.lr,
            betas= self.betas
        )

        # self.optimizer = torch.optim.Adam(
        #     self.policy.parameters(),
        #     lr = self.lr,
        #     betas= self.betas
        # )

        self.policy_old = ActorCritic(
            state_dim = num_inputs,
            action_dim = num_actions,
            n_latent_var = opt.n_latent_var
        ).to(opt.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):
        old_states = memory.states.detach()
        old_actions = memory.actions.detach()
        old_logprobs = memory.logprobs.detach()
        old_disreturn = memory.disreturn.detach()

        if old_disreturn.std() == 0:
            old_disreturn = (old_disreturn - old_disreturn.mean()) / 1e-5
        else:
            old_disreturn = (old_disreturn - old_disreturn.mean()) / old_disreturn.std()
        
        for epoch in range(self.k_epoches):
            logprobs, state_values, dis_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = old_disreturn - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss1 = -torch.min(surr1, surr2) - 0.005 * dis_entropy
            self.optimizer.zero_grad()
            loss1.mean().backward()
            self.optimizer.step()

            loss2 = self.MseLoss(state_values, old_disreturn)
            self.critic_optimizer.zero_grad()
            loss2.mean().backward()
            self.critic_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.mean().detach().cpu().numpy()
        
    


