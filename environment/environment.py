import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import math

import environment.cifar as models


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)

class Environment(torch.nn.Module):
    def __init__(self, config):
        super(Environment, self).__init__()

        self.config = config
        self.model = models.__dict__[config.model.arch](**config.model.kwargs)
        self.model.cuda()

        # define loss function (criterion) and optimizer
        self.criterion_train = nn.CrossEntropyLoss(reduction=self.config.loss_reduction)
        self.criterion_train_reference = nn.CrossEntropyLoss(reduction=self.config.loss_reduction)
        self.criterion_val = nn.CrossEntropyLoss()

        self.lr = self.config.e_learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                         momentum=self.config.momentum,
                                         weight_decay=self.config.weight_decay)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.e_learning_rate)
        self.reference_model = None
        if self.config.reward.reward_option == 'sub_reference_model':
            self.reference_model = models.__dict__[config.model.arch](**config.model.kwargs)
            self.reference_model.cuda()
            self.reference_optimizer = torch.optim.SGD(self.reference_model.parameters(), self.lr,
                                                       momentum=self.config.momentum,
                                                       weight_decay=self.config.weight_decay)

        cudnn.benchmark = True

        self.smoothed_training_loss = 2.3
        self.smoothed_validate_acc1 = 0.1
        self.smoothed_filter_rate = 0.
        self.stage = 0

        self.softmax = torch.nn.Softmax(dim=-1)

        self.reward_norm = math.exp(self.config.reward.reward_wk)

        self.baseline_rewards = None
        if self.config.baseline.baseline_in is not None:
            self.load_baseline()

    def load_baseline(self, smooth=False):
        # load rewards of baseline
        with open(self.config.baseline.baseline_in, 'r') as bf:
            tt = bf.readlines()
        self.baseline_rewards = np.zeros((len(tt) - 1) * self.config.eval_freq + 1, dtype=np.float32)
        anchors = []
        for i, rt in enumerate(tt):
            rt = float(rt.rstrip('\r\n'))
            anchors.append(rt)

        for i in range(self.baseline_rewards.shape[0] - 1):
            pre = i // self.config.eval_freq
            p = i % self.config.eval_freq
            v = anchors[pre] + p * (anchors[pre + 1] - anchors[pre]) / self.config.eval_freq
            self.baseline_rewards[i] = v

        if smooth:
            target = 0
            for i in range(self.baseline_rewards.shape[0]):
                target = target * (1 - self.config.b_tau) + self.baseline_rewards[i] * self.config.b_tau
                self.baseline_rewards[i] = target

    def forward(self):
        pass

    def weight_init(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_item_features(self, input_image):
        input_features = self.feature_model.features(input_image)
        input_features = input_features.view(input_features.size(0), -1)
        return input_features

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def reset(self):
        self.smoothed_training_loss = 2.3
        self.smoothed_validate_acc1 = 0.1
        self.smoothed_filter_rate = 0.
        self.stage = 0

        # initialize weights
        self.weight_init(self.model)
        # initialize optimizer
        self.lr = self.config.e_learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=self.config.momentum,
                                         weight_decay=self.config.weight_decay)
        if self.config.reward.reward_option == 'sub_reference_model':
            self.hard_update(self.reference_model, self.model)
            self.reference_optimizer = torch.optim.SGD(self.reference_model.parameters(), self.lr,
                                                       momentum=self.config.momentum,
                                                       weight_decay=self.config.weight_decay)

    def adjust_learning_rate_by_epoch(self, epoch, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        points = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.0001, 0.00001, 0.000001]
        self.lr = self.config.e_learning_rate * (points[epoch // 10])
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_learning_rate_by_step(self, step, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if step % self.config.val_step == 0:
            if step == (
                    self.config.train_num // self.config.num_candidates * 30) // self.config.val_step * self.config.val_step:
                lr_scale = 0.1
                self.lr = self.config.e_learning_rate * lr_scale
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

            if step == (
                    self.config.train_num // self.config.num_candidates * 40) // self.config.val_step * self.config.val_step:
                lr_scale = 0.01
                self.lr = self.config.e_learning_rate * lr_scale
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

            if step == (
                    self.config.train_num // self.config.num_candidates * 50) // self.config.val_step * self.config.val_step:
                lr_scale = 0.001
                self.lr = self.config.e_learning_rate * lr_scale
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

    def adjust_learning_rate_by_stage(self, step, num_stage_step, optimizer, lr_stages):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_scale = 0.1
        for i, s in enumerate(lr_stages):
            if step == num_stage_step * s:
                self.lr = self.config.e_learning_rate * np.power(lr_scale, i + 1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

        # if step == num_stage_step * 13:
        #     lr_scale = 0.01
        #     self.lr = self.config.e_learning_rate * lr_scale
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = self.lr
        #
        # if step == num_stage_step * 16:
        #     lr_scale = 0.001
        #     self.lr = self.config.e_learning_rate * lr_scale
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = self.lr

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def validate(self, model, val_loader, total_i):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):

                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                loss = self.criterion_val(output, target)

                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if total_i % self.config.print_freq == 0 and i % self.config.val_print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
        top1_acc = top1.avg.item()

        model.train()

        return top1_acc

    def train_step(self, input_pick, input_target, total_i, tb_logger, item_weights=None):
        # switch to train mode
        input_pick = self.model(input_pick)

        loss = self.criterion_train(input_pick, input_target)
        loss = loss.mean()
        prec1, prec5 = self.accuracy(input_pick, input_target, topk=(1, 5))
        self.smoothed_training_loss = self.config.smooth_rate * self.smoothed_training_loss + \
                                      (1 - self.config.smooth_rate) * loss.cpu().item()
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if total_i % self.config.print_freq == 0:
            if tb_logger is not None:
                tb_logger.add_scalar('loss_train', loss, total_i)
                tb_logger.add_scalar('acc1_train', prec1, total_i)
                tb_logger.add_scalar('acc5_train', prec5, total_i)
                tb_logger.add_scalar('lr', self.lr, total_i)
            print("law-->step:", total_i,
                  '\tstage:', self.stage,
                  "\tloss:", loss.item(),
                  "\ttop1:", prec1.item(),
                  "\ttop5:", prec5.item())

    def train_step_reference(self, input_pick, input_target, total_i, tb_logger, item_weights=None):
        # switch to train mode
        input_pick = self.reference_model(input_pick)

        loss = self.criterion_train_reference(input_pick, input_target)
        loss = loss.mean()
        prec1, prec5 = self.accuracy(input_pick, input_target, topk=(1, 5))

        self.reference_optimizer.zero_grad()
        loss.backward()
        self.reference_optimizer.step()

        if total_i % self.config.print_freq == 0:
            if tb_logger is not None:
                tb_logger.add_scalar('loss_train_reference', loss, total_i)
                tb_logger.add_scalar('acc1_train_reference', prec1, total_i)
                tb_logger.add_scalar('acc5_train_reference', prec5, total_i)
            print("ref-->step:", total_i,
                  '\tstage:', self.stage,
                  "\tloss_reference:", loss.item(),
                  "\ttop1_reference:", prec1.item(),
                  "\ttop5_reference:", prec5.item())

    def train_step_with_actor(self, input_pick, input_target, total_i, tb_logger, action, agent):

        input_pick = self.model(input_pick)

        loss = self.criterion_train(input_pick, input_target)
        item_weights = torch.ones_like(loss)
        loss_gain = 1.

        with torch.no_grad():
            loss_f = loss.clone().detach()
            loss_mean_no_filter = loss_f.mean()
            p = self.softmax(input_pick)

            f_list = []
            if self.config.features.use_loss:
                f_list = [loss_f.unsqueeze(dim=1)]
            print_list = [loss_f.unsqueeze(dim=1)]
            if self.config.features.use_loss_abs:
                loss_abs = loss_f - self.smoothed_training_loss
                print_list.append(loss_abs.unsqueeze(dim=-1))
                f_list.append(loss_abs.unsqueeze(dim=1))
            if self.config.features.use_loss_norm:
                loss_f_n = (loss_f - loss_f.mean()) / (loss_f.std() + 1e-5)
                print_list.append(loss_f_n.unsqueeze(dim=-1))
                f_list.append(loss_f_n.unsqueeze(dim=1))
            if self.config.features.use_logits:
                f_list.append(p.detach())
            if self.config.features.use_entropy:
                m = torch.distributions.categorical.Categorical(probs=p)
                e = m.entropy()
                if self.config.features.entropy_norm:
                    e = (e - e.mean()) / (loss_f.std() + 1e-5)
                f_list.append(e.unsqueeze(dim=1))
            if self.config.features.use_item_similarity:
                n = torch.nn.functional.normalize(input_pick)
                simi_mat = torch.matmul(n, n.transpose(0, 1))
                e = torch.eye(self.config.num_candidates).cuda()
                item_density = (simi_mat - e).mean(dim=-1, keepdim=True)
                f_list.append(item_density)
            if self.config.features.use_label:
                f_list.append(input_target.detach().unsqueeze(dim=1).float())

            if len(f_list) > 0:
                features = torch.cat(f_list, dim=1)
                item_weights = agent.weight_items(action, features)
            if total_i % self.config.print_freq == 0:
                print('action:', action)
                if len(f_list) > 0:
                    pass
                    # print("loss,...,item_weights")
                    # print_list.append(item_weights.unsqueeze(dim=-1))
                    # print_t = torch.cat(print_list, dim=-1)
                    # print(print_t)

                    # print("action:", action,
                    #       "\nitem_weights:", item_weights,
                    #       '\nloss', loss_f,
                    #       '\nloss_abs', loss_abs)

        loss *= item_weights
        loss = loss.sum() / item_weights.sum()

        loss_gap = loss.item() - loss_mean_no_filter.item()
        if self.config.learn_lr_gain:
            loss_gain = (1 + math.tanh(action[-1]))
            # loss_gain = max(0.2, min(action[-1], 10))
            loss *= loss_gain

        prec1, prec5 = self.accuracy(input_pick, input_target, topk=(1, 5))
        self.smoothed_training_loss = self.config.smooth_rate * self.smoothed_training_loss + \
                                      (1 - self.config.smooth_rate) * loss.cpu().item()

        # compute gradient and do SGD step
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        if total_i % self.config.print_freq == 0:
            if tb_logger is not None:
                tb_logger.add_scalar('loss_train', loss, total_i)
                tb_logger.add_scalar('acc1_train', prec1, total_i)
                tb_logger.add_scalar('acc5_train', prec5, total_i)
                tb_logger.add_scalar('lr', self.lr, total_i)
                tb_logger.add_scalar('loss_gap', loss_gap, total_i)
                tb_logger.add_scalar('loss_gain', loss_gain, total_i)
                tb_logger.add_scalar('stage', self.stage, total_i)

                if self.config.imbalance:
                    wn = item_weights.cpu().numpy()
                    ws = []
                    wo = []
                    for i in range(self.config.num_candidates):
                        ls = 10 if self.config.dataset == 'cifar10' else 100
                        if input_target[i] == ls - 1:
                            ws.append(wn[i])
                        if input_target[i] == 0:
                            wo.append(wn[i])
                    if len(ws) > 0:
                        ws_mean = np.mean(np.array(ws))
                        wo_mean = np.mean(np.array(wo))
                        tb_logger.add_scalar('ws_mean', ws_mean, total_i)
                        tb_logger.add_scalar('wo_mean', wo_mean, total_i)
                        print("step", total_i,
                              "\tws_mean:", ws_mean,
                              "\two_mean:", wo_mean)
            print("actor>step:", total_i,
                  '\tstage:', self.stage,
                  "\tloss:", loss.item(),
                  '\tloss_mean_no_filter:', loss_mean_no_filter.item(),
                  "\ttop1:", prec1.item(),
                  "\ttop5:", prec5.item(),
                  "\tloss_gain:", loss_gain)
        return item_weights.cpu().numpy(), loss_gap

    def step(self, input_batch, input_target, val_loader, total_i, tb_logger, action, agent, current_epoch,
             num_stage_step, done=False):
        item_weights, loss_gap = self.train_step_with_actor(input_batch, input_target, total_i, tb_logger, action,
                                                            agent)
        reward_1 = np.mean(item_weights)
        self.smoothed_filter_rate = self.config.smooth_rate * self.smoothed_filter_rate + (
                1 - self.config.smooth_rate) * reward_1
        reward_2 = 0.
        reward_3 = loss_gap
        reward_4 = np.linalg.norm(action, 2)

        if total_i > self.config.start_point:
            if (total_i + 1) % num_stage_step == 0 or done:
                acc1 = self.validate(self.model, val_loader, total_i)
                reward_2 = acc1
                self.smoothed_validate_acc1 = self.config.smooth_rate * self.smoothed_validate_acc1 \
                                              + (1 - self.config.smooth_rate) * acc1

                if self.baseline_rewards is not None and self.config.reward.reward_option == 'sub_baseline':
                    reward_2 = acc1 - self.baseline_rewards[total_i]
                elif self.config.reward.reward_option == 'sub_pre':
                    reward_2 = acc1 - self.smoothed_validate_acc1
                elif self.config.reward.reward_option == 'sub_reference_model':
                    acc1_reference = self.validate(self.reference_model, val_loader, total_i)
                    reward_2 = (acc1 - acc1_reference) / 10
                    if tb_logger is not None:
                        tb_logger.add_scalar('acc1_reference', acc1_reference, total_i)
                        tb_logger.add_scalar('acc1_relative', reward_2 * 10, total_i)

                if tb_logger is not None:
                    tb_logger.add_scalar('acc1', acc1, total_i)
                    tb_logger.add_scalar('reward_2', reward_2, total_i)
                    if self.baseline_rewards is not None:
                        tb_logger.add_scalar('smoothed_baseline', self.baseline_rewards[total_i], total_i)

        if self.config.reward.reward_weight:
            w = math.exp(
                self.config.reward.reward_wk * current_epoch / self.config.num_epoch_per_episode) / self.reward_norm * \
                self.config.reward.reward_scale
            reward_2 *= w
        reward = self.config.reward.filter_loss_rate * reward_1 + self.config.reward.acc_rate * reward_2 + \
                 self.config.reward.loss_gap_rate * reward_3 - self.config.reward.action_norm_rate * reward_4

        if tb_logger is not None and ((total_i + 1) % num_stage_step == 0 or done):
            tb_logger.add_scalar('reward', reward, total_i)
            tb_logger.add_scalar('item_weights_mean', reward_1, total_i)
            tb_logger.add_scalar('action_norm', reward_4, total_i)

        return reward
