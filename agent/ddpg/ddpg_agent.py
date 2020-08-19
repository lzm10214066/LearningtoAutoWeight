import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from agent.ddpg.actor_critic import (Actor, Critic)
from agent.ddpg.replay_buffer import ReplayBuffer
from agent.ddpg.random_process import OrnsteinUhlenbeckProcess

criterion = nn.MSELoss()


class DDPGAgent(nn.Module):
    def __init__(self, config):
        super(DDPGAgent, self).__init__()

        self.config = config
        self.actor = Actor(config)
        self.actor_target = Actor(config)

        self.critic = Critic(config)
        self.critic_target = Critic(config)

        self.actor_target.eval()
        self.critic_target.eval()

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)

        self.actor.cuda()
        self.critic.cuda()
        self.actor_target.cuda()
        self.critic_target.cuda()

        self.actor_optim = Adam(self.actor.parameters(), lr=self.config.a_learning_rate,
                                weight_decay=self.config.weight_decay)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.config.c_learning_rate,
                                 weight_decay=self.config.weight_decay)

        # Create replay buffer
        self.buffer = ReplayBuffer(self.config.buffer_size)

        # Hyper-parameters
        self.batch_size = self.config.policy_batch_size
        self.tau = self.config.tau
        self.discount = self.config.discount
        self.decay_epsilon = self.config.decay_epsilon
        self.epsilon = self.config.epsilon
        self.max_grad_norm = self.config.max_grad_norm

        self.is_training = True

        self.random_process = OrnsteinUhlenbeckProcess(size=self.config.action_dim, theta=self.config.ou_theta,
                                                       mu=self.config.ou_mu, sigma=self.config.ou_sigma)
        self.softmax = torch.nn.Softmax()

        self.norm_rate = self.config.actor_norm_rate

    def forward(self, *input):
        pass

    def update_actor_critic_full(self, total_i, tb_logger):
        if self.buffer.size() < self.batch_size:
            return

        print('replay_buffer_size:', self.buffer.size())
        for i in range(self.config.update_full_epoch):
            indices = np.arange(self.buffer.size())
            np.random.shuffle(indices)
            offset = 0
            while offset + self.config.policy_batch_size <= self.buffer.size():
                picked = indices[offset:offset + self.config.policy_batch_size]
                batch = [self.buffer.buffer[i] for i in picked]

                state_batch = np.array([_[0] for _ in batch])
                action_batch = np.array([_[1] for _ in batch])
                reward_batch = np.array([_[2] for _ in batch])
                next_state_batch = np.array([_[3] for _ in batch])
                terminal_batch = np.array([_[4] for _ in batch])

                state_batch = torch.from_numpy(state_batch).cuda()
                next_state_batch = torch.from_numpy(next_state_batch).cuda()
                # Prepare for the target q batch
                with torch.no_grad():
                    next_q_values = self.critic_target([
                        next_state_batch,
                        self.actor_target(next_state_batch),
                    ])

                target_q_batch = reward_batch + self.discount * (
                        1 - terminal_batch.astype(np.float32)) * next_q_values.squeeze().cpu().numpy()

                action_batch_t = torch.from_numpy(action_batch.astype(np.float32)).cuda()
                q_batch = self.critic([state_batch, action_batch_t])

                value_loss = criterion(q_batch, torch.tensor(target_q_batch, dtype=torch.float32).cuda())


                # Critic update
                self.critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                # Actor update
                actor_inter = self.actor(state_batch)
                policy_loss = -self.critic([
                    state_batch,
                    actor_inter
                ])
                actor_inter_norm = torch.norm(actor_inter, dim=-1)

                policy_loss = policy_loss.mean() + self.norm_rate*actor_inter_norm.mean()
                self.actor.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                # Target update
                self.soft_update(self.actor_target, self.actor, self.tau)
                self.soft_update(self.critic_target, self.critic, self.tau)

                # build summary
                offset += self.config.policy_batch_size
                if i == self.config.update_full_epoch - 1 and offset + self.config.policy_batch_size >= self.buffer.size():
                    tb_logger.add_scalar('critic loss', value_loss, total_i)
                    tb_logger.add_scalar('q', q_batch[0], total_i)
                    tb_logger.add_scalar('next_q', next_q_values[0], total_i)
                    tb_logger.add_scalar('epsilon', self.epsilon, total_i)

                    print("policy_step:", total_i,
                          "\tvalue_loss:", value_loss.item(),
                          "\tq_predict:", q_batch[0].item(),
                          "\tnext_q_predict:", next_q_values[0].item(),
                          "\tepsilon:", self.epsilon)

    def update_actor_critic(self, total_i, tb_logger, num_stage_step):
        if self.buffer.size() < self.batch_size:
            return
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = \
            self.buffer.sample_batch(self.batch_size)

        state_batch = torch.from_numpy(state_batch).cuda()
        next_state_batch = torch.from_numpy(next_state_batch).cuda()
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                next_state_batch,
                self.actor_target(next_state_batch),
            ])

        target_q_batch = reward_batch + self.discount * (1 - terminal_batch.astype(np.float32)) * next_q_values. \
            squeeze().cpu().numpy()

        action_batch_t = torch.from_numpy(action_batch.astype(np.float32)).cuda()
        q_batch = self.critic([state_batch, action_batch_t])

        value_loss = criterion(q_batch, torch.tensor(target_q_batch, dtype=torch.float32).cuda())

        # Critic update
        self.critic.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # Actor update
        policy_loss = -self.critic([
            state_batch,
            self.actor(state_batch)
        ])

        policy_loss = policy_loss.mean()
        self.actor.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        # build summary
        if total_i % num_stage_step == 0:
            tb_logger.add_scalar('critic loss', value_loss, total_i)
            tb_logger.add_scalar('q', q_batch[0], total_i)
            tb_logger.add_scalar('next_q', next_q_values[0], total_i)
            tb_logger.add_scalar('epsilon', self.epsilon, total_i)

            print("policy_step:", total_i,
                  "\tvalue_loss:", value_loss.item(),
                  "\tq_predict:", q_batch[0].item(),
                  "\tnext_q_predict:", next_q_values[0].item(),
                  "\tepsilon:", self.epsilon)

    def weight_items(self, action, items):
        with torch.no_grad():
            action = torch.from_numpy(action).unsqueeze(1).cuda()  # 512 -> 512
            offset = self.config.item_feature_dim * self.config.num_node
            w0 = action[0:offset, ...].view(self.config.item_feature_dim, self.config.num_node)
            b0 = action[offset:offset + 1, ...]

            scores = torch.matmul(items, w0) + b0

            if self.config.num_node > 1:
                offset += 1
                w1 = action[offset:offset + self.config.num_node, ...].view(self.config.num_node, 1)
                offset += self.config.num_node
                b1 = action[offset:offset + 1, ...]

                scores = scores.tanh()
                scores = torch.matmul(scores, w1) + b1

            if self.config.weight_option == 'bernoulli':
                p = scores.sigmoid().bernoulli()
            elif self.config.weight_option == 'classify':
                p = scores > 0.5
                if p.sum() < 10:
                    p = scores.sigmoid().bernoulli()
            elif self.config.weight_option == 'add_weight':
                p = 1 + scores.tanh()
                # p = scores.sigmoid()
        return p.float().squeeze()

    def select_action_with_explore(self, state, decay_epsilon=True):
        self.actor.eval()
        action = self.actor(state).cpu().numpy().squeeze(axis=0)
        e = np.random.random_sample()
        if e <= self.config.explore_prob:
            noise = self.random_process.sample()
            s = np.linalg.norm(action)
            action_s = action / s
            action_s += self.is_training * self.epsilon * noise
            action = action_s * s

        if decay_epsilon:
            self.epsilon = max(self.epsilon - self.config.decay_epsilon, 0)
        self.actor.train()
        return action

    def reset_noise(self):
        # self.s_t = obs
        self.random_process.reset_states()

    def load_model(self, path):
        if path is None:
            print('the path is None')
            return
        self.load_state_dict(torch.load(path))
        print('load ', path)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def seed(self, s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
