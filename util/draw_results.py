import numpy as np
import pylab as pl
# import matplotlib as plt
import matplotlib.pyplot as plt


# pl.style.use(plt.RcParams({
#     #'font.family': 'Times New Roman',
#     'font.size': 14,
#     'text.usetex': True,
# }))


def load_csv(file_path, name):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    if "Noisy" in name:
        data = data[:350, :]
    return data[:, 1], data[:, 2], name


def smooth(scalars, weight: float = 0.5):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


data_root = ''


def load_loss_gap():
    return [
        load_csv(data_root + '/imagenet/run_sampler_episode_24_tag_loss_gap.csv', "Loss_Gap_Episode_0"),
        load_csv(data_root + '/imagenet/run_sampler_episode_31_tag_loss_gap.csv', "Loss_Gap_Episode_1"),
        load_csv(data_root + '/imagenet/run_sampler_episode_40_tag_loss_gap.csv', "Loss_Gap Episode_2"),
    ]


def load_loss_gap_noise():
    return [
        load_csv(data_root + '/imagenet-noise/run_test_sampler_episode_0_tag_loss_gap.csv', "Loss_Gap_Noisy_Episode_0"),
        load_csv(data_root + '/imagenet-noise/run_test_sampler_episode_5_tag_loss_gap.csv', "Loss_Gap_Noisy_Episode_1"),
        load_csv(data_root + '/imagenet-noise/run_test_sampler_episode_10_tag_loss_gap.csv',
                 "Loss_Gap_Noisy_Episode_2"),
    ]


def load_weight_cifar10():
    return [
        load_csv(data_root + '/cifar10_imbalance_res18/run_test_sampler_episode_25_tag_wo_mean.csv',
                 "Weight_Mean_Others"),
        load_csv(data_root + '/cifar10_imbalance_res18/run_test_sampler_episode_25_tag_ws_mean.csv', "Weight_Mean_0")
    ]


def load_weight_cifar100():
    return [
        load_csv(data_root + '/cifar100_imbalance_res18/run_test_sampler_episode_20_tag_wo_mean.csv',
                 "Weight_Mean_Others"),
        load_csv(data_root + '/cifar100_imbalance_res18/run_test_sampler_episode_20_tag_ws_mean.csv', "Weight_Mean_0")
    ]


plot_list = [
    ["Loss_Gap", load_loss_gap(), 0.9],
    ["Loss_Gap_Noisy", load_loss_gap_noise(), 0.9],
    ["Weight_Mean_Cifar10", load_weight_cifar10(), 0.8],
    ["Weight_Mean_Cifar100", load_weight_cifar100(), 0.8]
]

for p_name, pairs, s in plot_list:
    labels = []

    for ((x, y, name), c) in zip(pairs, 'rgb'):
        labels.append(name)
        y_hat = smooth(y, s)
        pl.plot(x, y_hat, c)

    if s > 0:
        for ((x, y, name), c) in zip(pairs, 'rgb'):
            pl.plot(x, y, c + '--', alpha=0.1)

    pl.xlabel('Iterations')
    pl.ylabel(p_name)
    # pl.margins(0,0)
    pl.legend(labels)
    pl.grid()

    pl.savefig(f'./{p_name}.pdf', bbox_inches='tight')
    # pad_inches = 0)
    pl.show()
