import argparse
import yaml
from easydict import EasyDict
import os

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ddpg', help='algorithm to use: ddpg')

    parser.add_argument(
        '--config', default='./experiments/cifar10_imbalance_imbfactor-0.1/cifar10_imbalance_config.yaml', help='config')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    # add some other params
    if config.env.train_num < 0:
        config.env.train_num = config.env.num_classes_use * config.env.num_images_per_class

    config.env.num_step_per_epoch = config.env.train_num // config.env.num_candidates
    config.env.num_step_per_epoch_test = config.env.datasize // config.env.num_candidates

    config.env.num_stage_step = (config.env.num_step_per_epoch * config.env.num_epoch_per_episode) // config.env.num_stages
    config.env.num_stage_step_test = (config.env.num_step_per_epoch_test * config.env.num_epoch_per_episode) // config.env.num_stages

    config.env.num_warmup_step = config.env.num_warmup_stages * config.env.num_stage_step
    config.env.num_warmup_step_test = config.env.num_warmup_stages * config.env.num_stage_step_test

    config.agent.ddpg.item_feature_dim = 0
    if config.env.features.use_loss:
        config.agent.ddpg.item_feature_dim += 1
    if config.env.features.use_logits:
        config.agent.ddpg.item_feature_dim += 10
    if config.env.features.use_loss_norm:
        config.agent.ddpg.item_feature_dim += 1
    if config.env.features.use_loss_abs:
        config.agent.ddpg.item_feature_dim += 1
    if config.env.features.use_entropy:
        config.agent.ddpg.item_feature_dim += 1
    if config.env.features.use_item_similarity:
        config.agent.ddpg.item_feature_dim += 1
    if config.env.features.use_label:
        config.agent.ddpg.item_feature_dim += 1

    config.agent.ddpg.action_dim = 0
    if config.agent.ddpg.item_feature_dim > 0:
        config.agent.ddpg.action_dim = config.agent.ddpg.item_feature_dim + 1  # bias

    if config.env.learn_lr_gain:
        config.agent.ddpg.action_dim += 1
    if config.agent.ddpg.num_node > 1:
        config.agent.ddpg.action_dim = config.agent.ddpg.item_feature_dim * config.agent.ddpg.num_node + \
                                       config.agent.ddpg.num_node + 2

    config.agent.ddpg.state_dim = config.agent.ddpg.step_embedding_dim + config.agent.ddpg.other_state_dim
    config.agent.ddpg.num_stages = config.env.num_stages

    from agent.ddpg.runner import Runner

    config.log_root = os.path.dirname(args.config)

    runner = Runner(config)
    runner.run_ddpg()
