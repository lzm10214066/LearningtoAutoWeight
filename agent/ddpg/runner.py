import torch
from data.dataset import Cifar10, Cifar100
from environment.environment import Environment
from agent.ddpg.ddpg_agent import DDPGAgent
import collections

import os
import time
import numpy as np
import yaml

from tensorboardX import SummaryWriter
from data.util import EpisodeGivenSampler

from torch.utils.data import DataLoader

Transition = collections.namedtuple("Transition",
                                    ["state", "action", "reward", "next_state", "done"])


class Runner():
    def __init__(self, config):
        self.config = config

    def warm_up(self, env, input_image, target, episode_step, current_epoch, val_loader, tb_logger, b_out,
                num_stage_step):

        if episode_step % num_stage_step == 0:
            env.stage += 1

        if self.config.env.reward.reward_option == 'sub_reference_model':
            env.train_step_reference(input_image, target, episode_step, tb_logger)
        env.train_step(input_image, target, episode_step, tb_logger)

        if episode_step % self.config.env.eval_freq == 0:
            acc1 = env.validate(env.model, val_loader, episode_step)
            env.smoothed_validate_acc1 = self.config.env.smooth_rate * env.smoothed_validate_acc1 + (
                    1 - self.config.env.smooth_rate) * acc1
            if tb_logger is not None:
                tb_logger.add_scalar('acc1', acc1, episode_step)
                print('epoch:', current_epoch, '\nacc1:', acc1)

            if self.config.env.reward.reward_option == 'sub_reference_model':
                acc1_reference = env.validate(env.reference_model, val_loader, episode_step)
                if tb_logger is not None:
                    tb_logger.add_scalar('acc1_reference', acc1_reference, episode_step)
                    tb_logger.add_scalar('acc1_relative', acc1 - acc1_reference, episode_step)

            if b_out is not None:
                b_out.write(str(acc1) + '\n')

    def test_actor(self, agent, env, episode_step, input_image, target, test_loader, tb_logger, pre_buffer,
                   current_epoch, done):
        with torch.no_grad():
            action = pre_buffer[1]
            if action is None or episode_step % self.config.env.num_stage_step_test == 0 or done:
                env.stage += 1
                # env.smoothed_training_loss, env.smoothed_filter_rate, env.lr
                state = np.array([env.stage, env.smoothed_training_loss, env.smoothed_filter_rate, env.lr],
                                 dtype=np.float32)
                state_t = torch.from_numpy(np.expand_dims(state, axis=0)).cuda()
                action = agent.actor_target(state_t).cpu().squeeze(dim=0).numpy()

        if self.config.env.reward.reward_option == 'sub_reference_model':
            env.train_step_reference(input_image, target, episode_step, tb_logger)

        reward = env.step(input_image, target, test_loader, episode_step, tb_logger, action,
                          agent, current_epoch, self.config.env.num_stage_step_test, done)

        if (episode_step + 1) % self.config.env.num_stage_step_test == 0 or done:
            pre_buffer[2] = reward

        if episode_step % self.config.env.num_stage_step_test == 0:
            pre_buffer[0] = state
            pre_buffer[1] = action

        if (episode_step + 1) % self.config.env.num_stage_step_test == 0:
            acc_rl = env.validate(env.model, test_loader, episode_step)
            acc_reference = 0.
            if self.config.env.reward.reward_option == 'sub_reference_model':
                acc_reference = env.validate(env.reference_model, test_loader, episode_step)
            if tb_logger is not None:
                tb_logger.add_scalar('test_acc_rl', acc_rl, episode_step)
                tb_logger.add_scalar('test_acc_reference', acc_reference, episode_step)
                tb_logger.add_scalar('test_acc_relative', acc_rl - acc_reference, episode_step)

    def train_actor(self, agent, env, episode_step, input_image, target, val_loader, tb_logger, pre_buffer,
                    current_epoch, done):
        with torch.no_grad():
            action = pre_buffer[1]
            if action is None or episode_step % self.config.env.num_stage_step == 0 or done:
                env.stage += 1
                # env.smoothed_training_loss, env.smoothed_filter_rate, env.lr
                state = np.array([env.stage, env.smoothed_training_loss, env.smoothed_filter_rate, env.lr],
                                 dtype=np.float32)
                state_t = torch.from_numpy(np.expand_dims(state, axis=0)).cuda()
                action = agent.select_action_with_explore(state_t)

        if self.config.env.reward.reward_option == 'sub_reference_model':
            env.train_step_reference(input_image, target, episode_step, tb_logger)
        reward = env.step(input_image, target, val_loader, episode_step, tb_logger, action,
                          agent, current_epoch, self.config.env.num_stage_step, done)

        if self.config.agent.ddpg.update_full is False:
            agent.update_actor_critic(episode_step, tb_logger)

        if (episode_step + 1) % self.config.env.num_stage_step == 0 or done:
            pre_buffer[2] = reward

        if pre_buffer[0] is not None and (episode_step % self.config.env.num_stage_step == 0 or done):
            p_state = pre_buffer[0]
            p_action = pre_buffer[1]
            p_reward = pre_buffer[2]

            trans = Transition(state=p_state, action=p_action, next_state=state, reward=p_reward, done=done)
            agent.buffer.add(trans)

            if self.config.agent.ddpg.update_full:
                agent.update_actor_critic_full(episode_step, tb_logger)

        if episode_step % self.config.env.num_stage_step == 0:
            pre_buffer[0] = state
            pre_buffer[1] = action

    def run_ddpg(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

        # define model
        agent = DDPGAgent(self.config.agent.ddpg)
        agent.cuda()
        env = Environment(self.config.env)
        env.reset()

        if self.config.env.model_load_path is not None:
            agent.load_state_dict(torch.load(self.config.env.model_load_path))

        # define data
        if self.config.env.dataset == 'cifar100':
            dataset = Cifar100(self.config.env)
        else:
            dataset = Cifar10(self.config.env)

        train_actor_val_loader = DataLoader(
            dataset.train_actor_val_dataset, batch_size=self.config.env.val_batch_size, shuffle=False,
            num_workers=self.config.env.workers)
        test_actor_test_loader = DataLoader(
            dataset.test_actor_test_dataset, batch_size=self.config.env.val_batch_size, shuffle=False,
            num_workers=self.config.env.workers)

        # define log out
        time_array = time.localtime(time.time())
        log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time_array)
        # name log_dir with paras
        use_loss_norm = 'use_loss_norm' if self.config.env.features.use_loss_norm else 'no_loss_norm'
        use_logits = 'use_logits' if self.config.env.features.use_logits else 'no_logits'
        use_loss_abs = 'use_loss_abs' if self.config.env.features.use_loss_abs else 'no_loss_abs'
        use_loss_gain = 'use_loss_gain' if self.config.env.learn_lr_gain else 'no_loss_gain'
        log_dir_name = '_'.join([self.config.env.reward.reward_option, use_loss_norm, use_logits, use_loss_abs,
                                 str(self.config.env.reward.filter_loss_rate), str(self.config.agent.ddpg.buffer_size),
                                 str(self.config.agent.ddpg.a_learning_rate),
                                 str(self.config.agent.ddpg.c_learning_rate),
                                 self.config.agent.ddpg.weight_option, use_loss_gain, log_time])

        log_dir = os.path.join(self.config.log_root, self.config.env.log_dir, log_dir_name)
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)

        save_dir = os.path.join(self.config.log_root, self.config.env.save_dir, log_dir_name)
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        # parms out
        paras_path = os.path.join(log_dir, 'paras.yaml')
        with open(paras_path, "w", encoding='utf-8') as f:
            yaml.dump(self.config, f)

        b_out = None
        for current_episode in range(self.config.env.num_episode):
            episode_log = 'sampler_episode_' + str(current_episode)
            if (current_episode or self.config.env.model_load_path is not None) and current_episode % \
                    self.config.agent.ddpg.test_actor_step == 0:
                episode_log = 'test_sampler_episode_' + str(current_episode)
            log_path = os.path.join(log_dir, episode_log)
            tb_logger = SummaryWriter(log_path)

            # if self.config.env.baseline.baseline_out:
            #     if b_out is not None:
            #         b_out.close()
            #     b_out = open('baseline_episode_' + str(current_episode), 'w')

            agent.reset_noise()
            env.reset()
            episode_step = 0
            pre_buffer = [None, None, None]

            if current_episode and current_episode % self.config.env.save_interval == 0:
                save_path = os.path.join(save_dir, 'episode_' + str(current_episode) + '.pth')
                agent.save_model(save_path)

            if (current_episode or self.config.env.model_load_path is not None) and current_episode % \
                    self.config.agent.ddpg.test_actor_step == 0:
                sampler = EpisodeGivenSampler(dataset.test_actor_train_dataset, self.config.env.num_stages,
                                              self.config.env.num_stage_step_test, self.config.env.num_candidates,
                                              total_iters=self.config.env.total_iters)
                test_actor_train_loader = DataLoader(
                    dataset.test_actor_train_dataset, batch_size=self.config.env.num_candidates, shuffle=False,
                    num_workers=self.config.env.workers, drop_last=True, sampler=sampler)

                episode_done_step = len(test_actor_train_loader) - 1
                for i, (input_image, target) in enumerate(test_actor_train_loader):
                    env.adjust_learning_rate_by_stage(episode_step, self.config.env.num_stage_step_test, env.optimizer,
                                                      self.config.env.lr_stages)
                    if self.config.env.reward.reward_option == 'sub_reference_model':
                        env.adjust_learning_rate_by_stage(episode_step, self.config.env.num_stage_step_test,
                                                          env.reference_optimizer, self.config.env.lr_stages)

                    input_image = input_image.cuda()
                    target = target.cuda()

                    current_epoch = episode_step // self.config.env.num_step_per_epoch_test

                    if episode_step < self.config.env.num_warmup_step_test:
                        self.warm_up(env, input_image, target, episode_step, current_epoch, test_actor_test_loader,
                                     tb_logger, b_out, self.config.env.num_stage_step_test)
                    else:
                        done = (episode_step == episode_done_step)
                        self.test_actor(agent, env, episode_step, input_image, target,
                                        test_actor_test_loader, tb_logger, pre_buffer, current_epoch, done)
                    episode_step += 1
            else:
                sampler = EpisodeGivenSampler(dataset.train_actor_train_dataset, self.config.env.num_stages,
                                              self.config.env.num_stage_step, self.config.env.num_candidates,
                                              total_iters=self.config.env.total_iters)
                train_actor_train_loader = DataLoader(
                    dataset.train_actor_train_dataset, batch_size=self.config.env.num_candidates, shuffle=False,
                    num_workers=self.config.env.workers, drop_last=True, sampler=sampler)

                episode_done_step = len(train_actor_train_loader) - 1
                for i, (input_image, target) in enumerate(train_actor_train_loader):
                    env.adjust_learning_rate_by_stage(episode_step, self.config.env.num_stage_step, env.optimizer,
                                                      self.config.env.lr_stages)
                    if self.config.env.reward.reward_option == 'sub_reference_model':
                        env.adjust_learning_rate_by_stage(episode_step, self.config.env.num_stage_step,
                                                          env.reference_optimizer, self.config.env.lr_stages)

                    input_image = input_image.cuda()
                    target = target.cuda()

                    current_epoch = episode_step // self.config.env.num_step_per_epoch
                    if episode_step < self.config.env.num_warmup_step:
                        self.warm_up(env, input_image, target, episode_step, current_epoch, train_actor_val_loader,
                                     tb_logger, b_out, self.config.env.num_stage_step)
                    else:
                        done = episode_step == episode_done_step - 1
                        self.train_actor(agent, env, episode_step, input_image, target, train_actor_val_loader,
                                         tb_logger, pre_buffer, current_epoch, done)
                    episode_step += 1

        if b_out is not None:
            b_out.close()
