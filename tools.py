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
def dataset_split(train_images, train_labels, indep_noise, dep_noise, split_per=0.9, random_seed=1, num_classes=10):
    clean_train_labels = train_labels[:, np.newaxis]

    noisy_labels, actual_noise, transition_matrix = utils.noisify(clean_train_labels, train_images, indep_noise,
                                                                  dep_noise, num_classes)

    noisy_labels = noisy_labels.squeeze()

    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels, transition_matrix
