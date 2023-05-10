import os, torch
import os.path
import numpy as np
import torchvision.transforms as T
from math import inf
import torch.nn.functional as F


def col_norm(T):
    column_sums = np.sum(T, axis=0)
    T_norm = T / column_sums
    return T_norm


def row_norm(T):
    row_sums = np.sum(T, axis=1)
    T_norm = T / row_sums.reshape((-1, 1))
    return T_norm


def vector_norm(v):
    v_sum = np.sum(v)
    v_norm = v / v_sum
    return v_norm


def instance_independent_noisify(y, P, random_state=1):
    """
    Flip classes according to transition probability matrix T.
    random_state should be between 0 and the number of classes - 1.
    """

    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(y.shape[0]):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def instance_dependent_noisify(y, nb_classes):
    """
    Flip classes according to a randomly generated class distribution.
    random_state should be between 0 and the number of classes - 1.
    """
    new_y = y.copy()

    for idx in np.arange(y.shape[0]):

        flipper = np.random.RandomState(idx)  # Get a new seed for each sample
        flipped = flipper.multinomial(n=1, pvals=vector_norm(flipper.rand(1, nb_classes)[0]), size=1)
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_test(y_train, x_train, indep_noise, dep_noise, nb_classes=10):
    print(x_train.shape)

    P = np.ones((nb_classes, nb_classes))  # Generate True T
    P = (indep_noise / (nb_classes - 1)) * P
    P[0, 0] = 1. - indep_noise
    for i in range(1, nb_classes - 1):
        P[i, i] = 1. - indep_noise
    P[nb_classes - 1, nb_classes - 1] = 1. - indep_noise

    instance_dependent_index = int(y_train.shape[0]*dep_noise)
    # instance_independent_index = instance_dependent_index + int(y_train.shape[0]*indep_noise)

    y_train_dependent = instance_dependent_noisify(y_train[:instance_dependent_index], nb_classes)
    y_train_independent = instance_independent_noisify(y_train[instance_dependent_index:], P)

    y_train_noisy = np.concatenate((y_train_dependent, y_train_independent), axis=0)

    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print("ACTUAL NOISE RATE:", actual_noise)

    return y_train_noisy, actual_noise, P


def noisify(train_data, train_labels, seed, noise_rate, feature_size, percent_instance_noise, transform, num_class=10):
    np.random.seed(seed)

    # Creating some heavy noise rate for instance dependent
    q_ = np.random.normal(loc=(noise_rate + 0.4), scale=0.1, size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q) == 80000:
            break

    w = np.random.normal(loc=0, scale=1, size=(num_class, feature_size, num_class))
    d = np.random.normal(loc=0, scale=1, size=(num_class, num_class))
    P = np.ones((num_class, num_class))
    P = (noise_rate / (num_class - 1)) * P
    if noise_rate > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - noise_rate
        for i in range(1, num_class - 1):
            P[i, i] = 1. - noise_rate
        P[num_class - 1, num_class - 1] = 1. - noise_rate

    # w = np.random.normal(loc=0,scale=1,size=(32*32*3,num_class))
    TM = np.zeros((num_class, num_class))  # transition matrix constant
    noisy_labels = []
    flag_instance_indep_noise = []
    for i, sample_numpy in enumerate(train_data):
        # print('##########################################################')
        PILconv = T.ToPILImage()
        if feature_size == 28 * 28:
            sample_numpy = sample_numpy.reshape((28, 28))
        else:
            sample_numpy = sample_numpy.reshape((3, 32, 32))
        sample_tensor = torch.tensor(sample_numpy)
        sample_PIL = PILconv(sample_tensor)
        sample_tensor = transform(sample_PIL)
        sample_numpy = sample_tensor.numpy()
        sample = sample_numpy.flatten()
        flag_instance_noise = np.random.binomial(1, percent_instance_noise)
        if flag_instance_noise:
            p_all = np.matmul(sample, w[train_labels[i]])
            # print(p_all)
            p_all[train_labels[i]] = -inf
            # print(p_all)
            p_all = F.softmax(torch.tensor(p_all), dim=0)
            # print(p_all)
            p_all = q[i] * F.softmax(p_all).numpy()
            # print(p_all)
            p_all[train_labels[i]] = 1 - q[i]
            # print(p_all)
            p_all = p_all / sum(p_all)
            # print("Instance Dependent Noise")
            # print(p_all)
            # print("True class label")
            # print(train_labels[i])
            flag_instance_indep_noise.append(0)
        else:
            # Asymmetric noise
            # p_all = d[train_labels[i]].flatten()
            # p_all[train_labels[i]] = -inf
            # p_all = q[train_labels[i]]* F.softmax(torch.tensor(p_all),dim=0).numpy()
            # p_all[train_labels[i]] = 1 - q[train_labels[i]]
            # p_all = p_all/sum(p_all)

            p_all = P[:, train_labels[i]]
            TM[:, train_labels[i]] = p_all
            flag_instance_indep_noise.append(1)
        # print("Instance Independent Noise")
        # print(p_all)
        # print("True class label")
        # print(train_labels[i])

        noisy_labels.append(np.random.choice(np.arange(num_class), p=p_all))

    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum()) / train_data.shape[
        0]
    print("Overall noise rate")
    print(over_all_noise_rate)
    ind = torch.tensor(train_labels).eq(torch.tensor(noisy_labels))
    clean_data = train_data[ind]
    clean_lables = train_labels[ind]

    return noisy_labels, over_all_noise_rate, TM, flag_instance_indep_noise


def create_dir(args):
    save_dir = args.save_dir + '/' + args.dataset + '/' + args.loss_func + '/' + 'vol_min=' + args.vol_min + '/' + args.reg_type + '/' + 'q=%f' % args.q + '/' + 'k=%f' % args.k + '/' + '%s' % args.noise_type + '/' + 'uniform_noise_rate_%f' % (args.indep_noise_rate) + '/' + 'outlier_noise_rate_%f' % (args.dep_noise_rate) + '/' + 'lam=%f6_' % args.lam + '%d' % args.seed
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % (save_dir))

    model_dir = save_dir + '/models'

    if not os.path.exists(model_dir):
        os.system('mkdir -p %s' % (model_dir))

    matrix_dir = save_dir + '/matrix'

    if not os.path.exists(matrix_dir):
        os.system('mkdir -p %s' % (matrix_dir))

    logs = open(save_dir + '/log.txt', 'w')

    return save_dir, model_dir, matrix_dir, logs
