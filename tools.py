import numpy as np
from scipy.optimize import linear_sum_assignment
import utils


def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error


def get_estimation_error(T, T_true):
    row_ind, col_ind = linear_sum_assignment(-np.dot(np.transpose(T), T_true))
    T = T[:, col_ind]

    return np.sum(np.abs(T - T_true)) / np.sum(np.abs(T_true))


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, transform, input_size, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10, outlier_noise_rate=0.05):
    clean_train_labels = train_labels[:, np.newaxis]

    noisy_labels, real_noise_rate, transition_matrix, outliers = utils.noisify_multiclass_symmetric(clean_train_labels, train_images, input_size,
                                                noise=noise_rate, outlier_noise=outlier_noise_rate, transform=transform,
                                                random_state=random_seed, nb_classes=num_classes)

    noisy_labels = noisy_labels.squeeze()

    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    train_outliers, val_outliers = outliers[train_set_index], outliers[val_set_index]

    return train_set, val_set, train_labels, val_labels, transition_matrix, train_outliers, val_outliers
