import numpy as np
from scipy.optimize import linear_sum_assignment
import utils


def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm


def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error


def get_estimation_error(T, T_true):
    row_ind, col_ind = linear_sum_assignment(-np.dot(np.transpose(T), T_true))
    T = T[:, col_ind]

    return np.sum(np.abs(T - T_true)) / np.sum(np.abs(T_true))


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, noise_rate=0.5, percent_instance_noise=0.1, transform=[], split_per=0.9, random_seed=1, num_class=10, feature_size=28*28):

    noisy_labels, real_noise_rate, transition_matrix, flag_instance_dep_noise = utils.noisify(train_images,
                                                                                              train_labels,
                                                                                              random_seed,
                                                                                              noise_rate=noise_rate,
                                                                                              feature_size=feature_size,
                                                                                              percent_instance_noise=percent_instance_noise,
                                                                                              transform=transform,
                                                                                              num_class=num_class)

    noisy_labels = np.array(noisy_labels)
    flag_instance_dep_noise = np.array(flag_instance_dep_noise)

    noisy_labels = noisy_labels.squeeze()
    flag_instance_dep_noise = flag_instance_dep_noise.squeeze()

    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels_noisy, val_labels_noisy = noisy_labels[train_set_index], noisy_labels[val_set_index]
    clean_train_labels, clean_val_labels = train_labels[train_set_index], train_labels[val_set_index]
    flag_train, flag_val = flag_instance_dep_noise[train_set_index], flag_instance_dep_noise[val_set_index]

    return train_set, val_set, train_labels_noisy, val_labels_noisy, transition_matrix, clean_train_labels, clean_val_labels, flag_train, flag_val
