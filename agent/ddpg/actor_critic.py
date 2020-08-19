import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config

        self.fc0 = nn.Linear(self.config.state_dim, self.config.actor_net_width)
        # nn.init.xavier_normal_(self.fc0.weight, gain=0.1)
        nn.init.normal_(self.fc0.weight, 0, self.config.actor_init_scale)

        self.fc1 = nn.Linear(self.config.actor_net_width, self.config.actor_net_width * 2)
        # nn.init.xavier_normal_(self.fc1.weight, gain=0.1)
        nn.init.normal_(self.fc1.weight, 0, self.config.actor_init_scale)

        self.fc2 = nn.Linear(self.config.actor_net_width * 2, self.config.actor_net_width)
        # nn.init.xavier_normal_(self.fc2.weight, gain=0.1)
        nn.init.normal_(self.fc2.weight, 0, self.config.actor_init_scale)

        self.fc3 = nn.Linear(self.config.actor_net_width, self.config.action_dim)
        # nn.init.xavier_normal_(self.fc2.weight, gain=0.1)
        nn.init.normal_(self.fc3.weight, 0, self.config.actor_init_scale)

        self.embedding = nn.Embedding(self.config.num_stages + 10, self.config.step_embedding_dim)

    def forward(self, state):
        step = state[:, 0].long()
        net_state = state[:, 1:]
        x = self.embedding(step)
        x = torch.cat([x, net_state], dim=-1)

        x = self.fc0(x)
        x = x.relu_()

        x = self.fc1(x)
        x = x.relu_()

        x = self.fc2(x)
        x = x.relu_()

        x = self.fc3(x)
        # x = nn.functional.normalize(x, dim=-1)
        # x = x.tanh()
        return x


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()

        self.config = config

        self.fc0 = nn.Linear(self.config.state_dim + self.config.action_dim, self.config.critic_net_width)
        # nn.init.xavier_normal_(self.fc0.weight, gain=0.1)
        nn.init.normal_(self.fc0.weight, 0, 0.01)

        self.fc1 = nn.Linear(self.config.critic_net_width, self.config.critic_net_width * 2)
        # nn.init.xavier_normal_(self.fc1.weight, gain=0.1)
        nn.init.normal_(self.fc1.weight, 0, 0.01)

        self.fc2 = nn.Linear(self.config.critic_net_width * 2, self.config.critic_net_width)
        # nn.init.xavier_normal_(self.fc2.weight, gain=0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.01)

        self.fc3 = nn.Linear(self.config.critic_net_width, 1)
        # nn.init.xavier_normal_(self.fc3.weight, gain=0.1)
        nn.init.normal_(self.fc3.weight, 0, 0.01)
        self.embedding = nn.Embedding(self.config.num_stages + 10, self.config.step_embedding_dim)

    def forward(self, state_action):
        state, action = state_action
        step = state[:, 0].long()
        net_state = state[:, 1:]
        step_e = self.embedding(step)

        # action = nn.functional.normalize(action, dim=1)
        x = torch.cat([step_e, net_state, action], dim=1)
        x = self.fc0(x)
        x = x.relu_()
        x = self.fc1(x)
        x = x.relu_()
        x = self.fc2(x)
        x = x.relu_()
        q = self.fc3(x).squeeze()
        return q
