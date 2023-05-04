import os
import os.path
import numpy as np


def col_norm(T):
    column_sums = np.sum(T, axis=0)
    T_norm = T / column_sums
    return T_norm


def row_norm(T):
    row_sums = np.sum(T, axis=1)
    T_norm = T / row_sums.reshape((-1, 1))
    return T_norm


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
        flipped = flipper.multinomial(n=1, pvals=row_norm(flipper.rand(1, nb_classes)[0]), size=1)
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify(y_train, x_train, indep_noise, dep_noise, nb_classes=10):
    print(x_train.shape)

    P = np.ones((nb_classes, nb_classes))  # Generate True T
    P = (indep_noise / (nb_classes - 1)) * P
    P[0, 0] = 1. - indep_noise
    for i in range(1, nb_classes - 1):
        P[i, i] = 1. - indep_noise
    P[nb_classes - 1, nb_classes - 1] = 1. - indep_noise

    instance_dependent_index = int(y_train.shape[0]*dep_noise)
    instance_independent_index = instance_dependent_index + int(y_train.shape[0]*indep_noise)

    y_train_dependent = instance_dependent_noisify(y_train[:instance_dependent_index], nb_classes)
    y_train_independent = instance_independent_noisify(y_train[instance_dependent_index:instance_independent_index], P)

    y_train_noisy = np.concatenate((y_train_dependent, y_train_independent, y_train[instance_independent_index:]), axis=0)

    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print("ACTUAL NOISE RATE:", actual_noise)

    return y_train_noisy, actual_noise, P


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
