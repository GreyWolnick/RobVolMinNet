import os, torch
import os.path
import copy, pdb
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal

from models import Outlier


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    #    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1

        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def norm(T):
    column_sums = np.sum(T, axis=0)
    T_norm = T / column_sums
    return T_norm

def multiclass_outlier_noisify(x, y, transform, nb_classes=10, random_state=1):
    """
        adds gross outliers to training labels
    """

    outlier = Outlier(784, 200, nb_classes)  # make these non-static
    unflatten = torch.nn.Unflatten(0, (nb_classes, nb_classes))
    flipper = np.random.RandomState(random_state)

    new_y = y.copy()

    # print(x.shape)
    # print(x[1].shape)
    #
    # print("Original Image:", transform(x[1]), "\n")
    # print("Flattened Image:", torch.flatten(transform(x[1])), "\n")
    # outlier = outlier(torch.flatten(transform(x[1])))
    # print(outlier)

    for idx in np.arange(x.shape[0]):
        i = y[idx]

        sample_T = unflatten(outlier(torch.flatten(transform(x[idx])))).cpu().detach().numpy()
        sample_T = norm(sample_T)  # Issue: only produces really low values

        if idx % 1000 == 0:
            print(sample_T)

        print("PRINTGIN", sample_T[i, :])
        print("2", sample_T[i, :][0])

        flipped = flipper.multinomial(1, sample_T[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_multiclass_symmetric(y_train, x_train, noise, outlier_noise, transform, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        # change the 2 below this
        sample_idx = np.random.choice(x_train.shape[0], 2, replace=False)
        # how do I want to split these labels so outliers are not used in multiclass_noisify
        y_train_outlier = multiclass_outlier_noisify(x_train[sample_idx, :], y_train[sample_idx, :], transform=transform,
                                                     nb_classes=nb_classes, random_state=random_state)
        print(y_train_outlier)
        exit()
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate, t = noisify_multiclass_symmetric(train_labels, noise_rate,
                                                                                random_state=random_state,
                                                                                nb_classes=nb_classes)

    return train_noisy_labels, actual_noise_rate


def create_dir(args):
    save_dir = args.save_dir + '/' + args.dataset + '/' + args.loss_func + '/' + 'q=%f' % args.q + '/' + 'k=%f' % args.k + '/' + '%s' % args.noise_type + '/' + 'noise_rate_%s' % args.noise_rate + '/' + 'lam=%f6_' % args.lam + '%d' % args.seed
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
