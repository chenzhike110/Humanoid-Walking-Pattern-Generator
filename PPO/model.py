import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, N_S, N_A):
        super().__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.sigma = nn.Linear(64,N_A)
        self.mu = nn.Linear(64,N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        self.set_init([self.fc1,self.fc2, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal
    
    def set_init(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias,0.)
    
    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))

        mu = self.mu(x)
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu, sigma
    
    def choose_action(self, s, evaluate=False):
        mu,sigma = self.forward(s)
        if evaluate:
            action = mu.max(0)
        else:
            Pi = self.distribution(mu,sigma)
            action = Pi.sample().numpy()
        return action

class Critic(nn.Module):
    def __init__(self,N_S):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values