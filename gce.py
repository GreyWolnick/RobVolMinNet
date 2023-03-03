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

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--q', type=float, help='q parameter in gce', default=0.7)
parser.add_argument('--k', type=float, help='k parameter in gce', default=0.5)
parser.add_argument('--save_dir', type=str, help='dir to save model files', default='saves')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--loss_func', type=str, default='gce')
parser.add_argument('--vol_min', type=bool, default=True)
parser.add_argument('--noise_type', type=str, default='symmetric')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--uniform_noise_rate', type=float, help='uniform corruption rate, should be less than 1', default=0.2)
parser.add_argument('--outlier_noise_rate', type=float, help='outlier corruption rate, should be less than 1', default=0.05)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--lam', type=float, default=0.0001)
parser.add_argument('--anchor', action='store_false')

parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--start_prune', default=40, type=int, help='number of total epochs to run')

args = parser.parse_args()
np.set_printoptions(precision=2, suppress=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# GPU
device = torch.device('cuda:' + str(args.device))

if args.dataset == 'mnist':
    args.n_epoch = 60
    num_classes = 10
    milestones = None

    train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         uniform_noise_rate=args.uniform_noise_rate, outlier_noise_rate=args.outlier_noise_rate,
                                         random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)

    val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       uniform_noise_rate=args.uniform_noise_rate, outlier_noise_rate=args.outlier_noise_rate,
                                       random_seed=args.seed, noise_type=args.noise_type)

    test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = Lenet()
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

if args.dataset == 'fashionmnist':
    args.n_epoch = 60
    num_classes = 10
    milestones = None

    train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                         uniform_noise_rate=args.uniform_noise_rate, outlier_noise_rate=args.outlier_noise_rate,
                                         random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
    val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                       uniform_noise_rate=args.uniform_noise_rate, outlier_noise_rate=args.outlier_noise_rate,
                                       random_seed=args.seed, noise_type=args.noise_type)
    test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    model = Lenet()
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

if args.dataset == 'cifar10':
    args.n_epoch = 80

    args.num_classes = 10
    milestones = [30, 60]

    train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                           uniform_noise_rate=args.uniform_noise_rate, outlier_noise_rate=args.outlier_noise_rate,
                                           random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
    val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                         uniform_noise_rate=args.uniform_noise_rate, outlier_noise_rate=args.outlier_noise_rate,
                                         random_seed=args.seed, noise_type=args.noise_type)
    test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset),
                                               target_transform=transform_target)
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
                                            noise_rate=args.noise_rate,
                                            random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
    val_data = data_load.cifar100_dataset(False, transform=transform_test(args.dataset),
                                          target_transform=transform_target,
                                          noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    test_data = data_load.cifar100_test_dataset(transform=transform_test(args.dataset),
                                                target_transform=transform_target)
    model = ResNet34(args.num_classes)
    trans = sig_t(device, args.num_classes, init=args.init)
    optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

save_dir, model_dir, matrix_dir, logs = create_dir(args)

print(args, file=logs, flush=True)

# optimizer and StepLR
optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
# optimizer_es = optim.Adam(model.parameters(), lr=args.lr) # Learning rate probably too high for MNIST and FASHION
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

best_acc = 0

print(train_data.t, file=logs, flush=True)

t = trans()
est_T = t.detach().cpu().numpy()
print(est_T, file=logs, flush=True)

estimate_error = tools.error(est_T, train_data.t)

print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)


def checkpoint(acc, epoch, net):
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
    torch.save(state, './checkpoint/ckpt.t7.' +
               args.sess)


for epoch in range(args.n_epoch):
    print('epoch {}'.format(epoch), file=logs, flush=True)

    model.train()
    trans.train()

    train_loss = 0.
    train_vol_loss = 0.
    train_acc = 0.
    val_loss = 0.
    val_acc = 0.
    test_loss = 0.
    test_acc = 0.

    if args.loss_func == "gce" and (epoch + 1) >= args.start_prune and (epoch + 1) % 10 == 0:
        checkpoint_dict = torch.load('./checkpoint/ckpt.t7.' + args.sess)
        model = checkpoint_dict['net']
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            indexes = [i for i in range(0, len(inputs))]
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            criterion.update_weight(outputs, targets, indexes)
        now = torch.load('./checkpoint/current_net')
        model = now['current_net']
        model.train()

    loss_list = []

    for batch_idx, (inputs, targets, indexes) in enumerate(train_loader):
        print(indexes)
        exit()
        # indexes = [i for i in range(0, len(inputs))]
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer_es.zero_grad()
        optimizer_trans.zero_grad()

        clean = model(inputs)

        t = trans()

        out = torch.mm(clean, t)

        vol_loss = t.slogdet().logabsdet

        if args.loss_func == "gce":
            ce_loss = criterion(out.log(), targets, indexes)
        else:
            ce_loss = criterion(out.log(), targets.long())

        loss = ce_loss + args.lam * vol_loss

        loss_list

        train_loss += loss.item()
        train_vol_loss += vol_loss.item()

        pred = torch.max(out, 1)[1]
        train_correct = (pred == targets).sum()
        train_acc += train_correct.item()

        loss.backward()
        optimizer_es.step()
        optimizer_trans.step()

    print(
        'Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / (len(train_data)) * args.batch_size,
                                                                   train_vol_loss / (
                                                                       len(train_data)) * args.batch_size,
                                                                   train_acc / (len(train_data))), file=logs,
        flush=True)

    scheduler1.step()
    scheduler2.step()

    with torch.no_grad():  # No actual hyperparameter testing, no use of lambda
        model.eval()
        trans.eval()

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            indexes = [i for i in range(0, len(inputs))]
            inputs, targets = inputs.cuda(), targets.cuda()

            clean = model(inputs)
            t = trans()

            out = torch.mm(clean, t)

            if args.loss_func == "gce":
                loss = criterion(clean, targets, indexes)
            else:
                loss = criterion(out.log(), targets.long())

            val_loss += loss.item()
            pred = torch.max(out, 1)[1]
            val_correct = (pred == targets).sum()
            val_acc += val_correct.item()

    print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data)) * args.batch_size,
                                                 val_acc / (len(val_data))), file=logs, flush=True)

    with torch.no_grad():
        model.eval()
        trans.eval()

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            indexes = [i for i in range(0, len(inputs))]
            inputs, targets = inputs.cuda(), targets.cuda()

            clean = model(inputs)

            if args.loss_func == "gce":
                loss = criterion(clean, targets, indexes)
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
        estimate_error = tools.error(est_T, train_data.t)

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

val_loss_array = np.array(val_loss_list)
val_acc_array = np.array(val_acc_list)
model_index = np.argmin(val_loss_array)
model_index_acc = np.argmax(val_acc_array)

matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index + 1)
final_est_T = np.load(matrix_path)
final_estimate_error = tools.error(final_est_T, train_data.t)

matrix_path_acc = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index_acc + 1)
final_est_T_acc = np.load(matrix_path_acc)
final_estimate_error_acc = tools.error(final_est_T_acc, train_data.t)

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
logs.close()
