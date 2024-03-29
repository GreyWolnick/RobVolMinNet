import tools
from utils import create_dir
import data_load
import argparse
from models import *
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import transform_train, transform_test, transform_target
from torch.optim.lr_scheduler import MultiStepLR
from truncatedloss import TruncatedLoss
from loss import symmetric_cross_entropy

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--q', type=float, help='q parameter in gce', default=0.5)
parser.add_argument('--k', type=float, help='k parameter in gce', default=0.4)
parser.add_argument('--save_dir', type=str, help='dir to save model files', default='saves')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--loss_func', type=str, default='sl')
parser.add_argument('--reg_type', type=str, default='min')
parser.add_argument('--vol_min', type=str, default='True')
# parser.add_argument('--vol_min', action='store_true') Possibly?
parser.add_argument('--noise_type', type=str, default='symmetric')  # Get rid of this
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)
parser.add_argument('--percent_instance_noise', type = float, default =0.1)
parser.add_argument('--indep_noise_rate', type=float, help='instance independent corruption rate, should be less than 1', default=0.2)
parser.add_argument('--dep_noise_rate', type=float, help='instance dependent corruption rate, should be less than 1', default=0.4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--lam', type=float, default=0.00001)
parser.add_argument('--anchor', action='store_false')  # Seems unnecessary

parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--start_prune', default=40, type=int, help='number of total epochs to run')

parser.add_argument('--alpha', default=0.1, type=float, help='alpha parameter for SL')
parser.add_argument('--beta', default=1.0, type=float, help='beta parameter for SL')

args = parser.parse_args()
np.set_printoptions(precision=2, suppress=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# GPU
device = torch.device('cuda:' + str(args.device))

if args.vol_min != 'True':  # Remove any volume regularization
    args.lam = 0

if args.dataset == 'mnist':
    args.n_epoch = 60
    num_classes = 10
    milestones = None

    train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset),
                                         target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed, anchor=args.anchor)
    val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                       random_seed=args.seed)
    test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = Lenet()
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

if args.dataset == 'fashionmnist':
    args.n_epoch = 60
    num_classes = 10
    milestones = None

    train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset),
                                         target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed, anchor=args.anchor)
    val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                       random_seed=args.seed)
    test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = Lenet()
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

if args.dataset == 'cifar10':
    args.n_epoch = 80

    args.num_classes = 10
    milestones = [30, 60]

    train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset),
                                         target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed, anchor=args.anchor)

    val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed)
    test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = ResNet18(args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)

if args.dataset == 'cifar100':
    args.init = 4.5
    args.n_epoch = 80

    args.num_classes = 100

    milestones = [30, 60]

    train_data = data_load.cifar100_dataset(True, transform=transform_train(args.dataset),
                                         target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed, anchor=args.anchor)

    val_data = data_load.cifar100_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed)
    test_data = data_load.cifar100_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)

    model = ResNet34(args.num_classes)
    trans = sig_t(device, args.num_classes, init=args.init)
    optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

if args.dataset == 'clothing1m':
    args.n_epoch = 80

    args.num_classes = 10
    milestones = [30, 60]

    train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset),
                                         target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed, anchor=args.anchor)

    val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                         noise_rate=args.noise_rate, percent_instance_noise=args.percent_instance_noise,
                                         random_seed=args.seed)
    test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = ResNet18(args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)

save_dir, model_dir, matrix_dir, logs = create_dir(args)

print(args, file=logs, flush=True)

# optimizer and StepLR
optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)
scheduler2 = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)

# data_loader NORMAL
train_loader = DataLoader(dataset=train_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=False)

val_loader = DataLoader(dataset=val_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        drop_last=False)

test_loader = DataLoader(dataset=test_data,
                         batch_size=args.batch_size,
                         num_workers=4,
                         drop_last=False)

if args.loss_func == "gce":
    criterion = TruncatedLoss(args.q, args.k, trainset_size=len(train_data)).cuda()  # Truncated Loss
elif args.loss_func == "sl":
    criterion = symmetric_cross_entropy(args.alpha, args.beta)
else:
    criterion = F.nll_loss  # Negative Log Likelihood Loss

# cuda
if torch.cuda.is_available:
    model = model.to(device)
    trans = trans.to(device)

train_loss_list = []
train_acc_list = []

val_loss_list = []
val_acc_list = []

test_acc_list = []
test_loss_list = []

t_est_error_list = []
t_vol_list = []

outlier_detection_rate_list = []

best_acc = 0
best_val_acc = 0

print(train_data.t, file=logs, flush=True)


t = trans()
est_T = t.detach().cpu().numpy()
print(est_T, file=logs, flush=True)

# estimate_error = tools.error(est_T, train_data.t)
estimate_error = tools.get_estimation_error(est_T, train_data.t)

print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)


def checkpoint(acc, epoch, net, type="gce"):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7.' + type)

def maximum_volume_regularization(H):
    HH = torch.mm(H.t(), H)
    regularizer_loss = -torch.log(torch.linalg.det(HH))
    return regularizer_loss

def minimum_volume_regularization(T):
    return T.slogdet().logabsdet


for epoch in range(args.n_epoch):
    print('epoch {}'.format(epoch), file=logs, flush=True)
    print(f'epoch {epoch}')

    # if args.loss_func == "gce" and (epoch + 1) % 10 == 0:
    #     fig, ax = plt.subplots()
    #     ax.plot(criterion.get_weight())
    #     ax.set_title('w')
    #     ax.set_xlabel('Index')
    #     fig.savefig(f'epoch{epoch}.png')
    #     # outlier_detection_rate = (train_data.train_outliers != criterion.get_weight()).mean()
    #     # outlier_detection_rate_list.append(outlier_detection_rate)

    model.train()
    trans.train()

    train_loss = 0.
    train_vol_loss = 0.
    train_acc = 0.
    val_loss = 0.
    val_acc = 0.
    test_loss = 0.
    test_acc = 0.
    weight_acc = 0.

    if args.loss_func == "gce" and (epoch + 1) >= args.start_prune and (epoch + 1) % 10 == 0:
        # checkpoint_dict = torch.load('./checkpoint/ckpt.t7.' + args.sess)
        # model = checkpoint_dict['net']
        # model.eval()
        # for batch_idx, (inputs, targets, indexes) in enumerate(train_loader):
        #     inputs, targets = inputs.cuda(), targets.cuda()
        #     clean = model(inputs)
        #     t = trans()
        #     out = torch.mm(clean, t)
        #     if args.vol_min != 'True':  # Revert T correction if vol_min is False
        #         out = clean
        #     criterion.update_weight(out, targets, indexes)
        # now = torch.load('./checkpoint/current_net')
        # model = now['current_net']
        # model.train()

        print('Pruning')
        checkpoint_dict = torch.load('./checkpoint/ckpt.t7.gce')
        model = checkpoint_dict['net']
        model.eval()
        for batch_idx, (inputs, targets, targets_clean, indexes, flag_noise_type) in enumerate(train_loader):
            inputs, targets, targets_clean = inputs.to(device), targets.to(device), targets_clean.to(device)
            clean = model(inputs)
            t = trans()
            out = torch.mm(clean, t)
            criterion.update_weight(out, targets, indexes, args.device, flag_noise_type)
            weights = criterion.get_weight(indexes)
            flag_clean = torch.eq(targets, targets_clean)
            flag_clean = flag_clean.to(args.device, dtype=torch.int32)
            flag_noise_type = flag_noise_type.to(args.device, dtype=torch.int32)
            flag_sel = torch.eq(flag_clean, flag_noise_type)
            flag_sel = flag_sel.to(args.device, dtype=torch.int32)
            weights = weights.to(args.device, dtype=torch.int32)
            # print('##################')
            # print(flag_noise_type)
            # print(weights)
            weight_correct = (flag_sel == weights).sum()
            weight_acc += weight_correct.item()

        print('Weight Acc: {:.6f}'.format(weight_acc / (len(train_data))), file=logs, flush=True)
        print('Weight Acc: {:.6f}'.format(weight_acc / (len(train_data))))

        now = torch.load('./checkpoint/current_net')
        model = now['current_net']
        model.train()

    for batch_idx, (inputs, targets, targets_clean, indexes, flag_noise_type) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_es.zero_grad()
        optimizer_trans.zero_grad()

        clean = model(inputs)

        t = trans()

        out = torch.mm(clean, t)

        if args.vol_min != 'True':  # Revert T correction if vol_min is False
            out = clean

        if args.reg_type == "max":
            regularizer_loss = maximum_volume_regularization(clean)
        else:
            regularizer_loss = minimum_volume_regularization(t)

        if np.isnan(regularizer_loss.item()) or np.isinf(regularizer_loss.item()) or regularizer_loss.item() > 100:
            regularizer_loss = torch.tensor(0.0)  # Fix?

        if args.loss_func == "gce":
            # out.log()
            ce_loss = criterion(out, targets, indexes)
        elif args.loss_func == "sl":
            ce_loss = criterion(torch.nn.functional.one_hot(targets, args.num_classes).cpu(), out.cpu())
        else:
            ce_loss = criterion(out.log(), targets.long())

        loss = ce_loss + args.lam * regularizer_loss

        train_loss += loss.item()
        train_vol_loss += regularizer_loss.item()

        pred = torch.max(out, 1)[1]
        train_correct = (pred == targets).sum()
        train_acc += train_correct.item()

        loss.backward()
        optimizer_es.step()
        optimizer_trans.step()

    print(
        'Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / (len(train_data)) * args.batch_size,
                                                                   train_vol_loss / (len(train_data)) * args.batch_size,
                                                                   train_acc / (len(train_data))), file=logs, flush=True)

    scheduler1.step()
    scheduler2.step()

    with torch.no_grad():  # No actual hyperparameter testing, no use of lambda
        model.eval()
        trans.eval()

        for batch_idx, (inputs, targets, targets_clean, indexes, flag_noise_type) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            clean = model(inputs)

            t = trans()
            out = torch.mm(clean, t)

            if args.vol_min != 'True':  # Revert T correction if vol_min is False
                out = clean

            if args.loss_func == "gce":
                loss = criterion(out, targets, indexes)  # Clean or out????
            elif args.loss_func == "sl":
                ce_loss = criterion(torch.nn.functional.one_hot(targets, args.num_classes), out)
            else:
                loss = criterion(out.log(), targets.long())

            val_loss += loss.item()
            pred = torch.max(out, 1)[1]
            val_correct = (pred == targets).sum()
            val_acc += val_correct.item()

        acc = val_acc / (len(val_data))
        if acc > best_val_acc:
            best_val_acc = acc
            checkpoint(acc, epoch, model, "val")

    print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data)) * args.batch_size,
                                                 val_acc / (len(val_data))), file=logs, flush=True)

    with torch.no_grad():
        model.eval()
        trans.eval()

        for batch_idx, (inputs, targets, indexes) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            clean = model(inputs)

            if args.loss_func == "gce":
                loss = criterion(clean, targets, indexes)
            elif args.loss_func == "sl":
                ce_loss = criterion(torch.nn.functional.one_hot(targets, args.num_classes), clean)
            else:
                loss = criterion(clean.log(), targets.long())

            test_loss += loss.item()
            pred = torch.max(clean, 1)[1]
            eval_correct = (pred == targets).sum()
            test_acc += eval_correct.item()

        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss / (len(test_data)) * args.batch_size,
                                                      test_acc / (len(test_data))), file=logs, flush=True)

        # Save checkpoint.
        acc = test_acc / (len(test_data))
        if acc > best_acc:
            best_acc = acc
            checkpoint(acc, epoch, model)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        state = {
            'current_net': model,
        }
        torch.save(state, './checkpoint/current_net')

        est_T = t.detach().cpu().numpy()
        estimate_error = tools.get_estimation_error(est_T, train_data.t)

        matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (epoch + 1)
        np.save(matrix_path, est_T)

        print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)
        print(est_T, file=logs, flush=True)
        t_est_error_list.append(estimate_error)
        t_vol_list.append(train_vol_loss / (len(train_data)) * args.batch_size)

    train_loss_list.append(train_loss / (len(train_data)) * args.batch_size)
    train_acc_list.append(train_acc / (len(train_data)))
    val_loss_list.append(val_loss / (len(val_data)) * args.batch_size)
    val_acc_list.append(val_acc / (len(val_data)))
    test_loss_list.append(test_loss / (len(test_data)) * args.batch_size)
    test_acc_list.append(test_acc / (len(test_data)))


checkpoint_dict = torch.load('./checkpoint/ckpt.t7.val')  # Get best val test accuracy
model = checkpoint_dict['net']
model.eval()
final_loss, final_acc = 0, 0
for batch_idx, (inputs, targets, indexes) in enumerate(test_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    clean = model(inputs)

    if args.loss_func == "gce":
        loss = criterion(clean, targets, indexes)
    elif args.loss_func == "sl":
        loss = criterion(torch.nn.functional.one_hot(targets, args.num_classes), clean)
    else:
        loss = criterion(clean.log(), targets.long())

    final_loss += loss.item()
    pred = torch.max(clean, 1)[1]
    final_correct = (pred == targets).sum()
    final_acc += final_correct.item()

print('Best Model Test Loss: {:.6f}, Acc: {:.6f}'.format(final_loss / (len(test_data)) * args.batch_size,
                                              final_acc / (len(test_data))), file=logs, flush=True)

val_loss_array = np.array(val_loss_list)
val_acc_array = np.array(val_acc_list)
model_index = np.argmin(val_loss_array)
model_index_acc = np.argmax(val_acc_array)

matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index + 1)
final_est_T = np.load(matrix_path)
# final_estimate_error = tools.error(final_est_T, train_data.t)
final_estimate_error = tools.get_estimation_error(final_est_T, train_data.t)

matrix_path_acc = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index_acc + 1)
final_est_T_acc = np.load(matrix_path_acc)
# final_estimate_error_acc = tools.error(final_est_T_acc, train_data.t)
final_estimate_error_acc = tools.get_estimation_error(final_est_T_acc, train_data.t)

print("Final test accuracy: %f" % test_acc_list[model_index], file=logs, flush=True)
print("Final test accuracy acc: %f" % test_acc_list[model_index_acc], file=logs, flush=True)
print("Final estimation error loss: %f" % final_estimate_error, file=logs, flush=True)
print("Final estimation error loss acc: %f" % final_estimate_error_acc, file=logs, flush=True)
print("Best epoch: %d" % model_index, file=logs, flush=True)
print(final_est_T, file=logs, flush=True)

print("Summary Metrics:", file=logs, flush=True)
print("T Accuracy:", t_est_error_list, file=logs, flush=True)
print("T Loss:", t_vol_list, file=logs, flush=True)
print("Training Loss:", train_loss_list, file=logs, flush=True)
print("Training Accuracy:", train_acc_list, file=logs, flush=True)
print("Testing Loss:", test_loss_list, file=logs, flush=True)
print("Testing Accuracy:", test_acc_list, file=logs, flush=True)
print("Outlier Detection:", outlier_detection_rate_list, file=logs, flush=True)
logs.close()
