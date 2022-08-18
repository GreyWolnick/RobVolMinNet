import tools
import data_load
import argparse
from models import *
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import transform_train, transform_test, transform_target
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--save_dir', type=str, help='dir to save model files', default='saves')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_type', type=str, default='symmetric')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--lam', type=float, default=0.0001)
parser.add_argument('--anchor', action='store_false')

args = parser.parse_args()
np.set_printoptions(precision=2, suppress=True)

# torch.backends.cudnn.benchmark = True  Necessary??
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# CPU
device = torch.device("cpu")

# # GPU
# device = torch.device('cuda:' + str(args.device))

args.n_epoch = 60
num_classes = 10
milestones = []  # Was None

train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset),
                                     target_transform=transform_target,
                                     noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type,
                                     anchor=args.anchor)
val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                   noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
model = Network()
# model = Lenet()


save_dir, model_dir, matrix_dir, logs = create_dir(args)

print(args, file=logs, flush=True)

# optimizer and StepLR
optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)
# scheduler2 = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)

# data_loader
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

loss_func_ce = F.nll_loss


def main():
    for epoch in range(2):  # loop over the dataset multiple times  OLD MNIST CODE

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer_es.zero_grad()  # Which optimizer to use in training

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func_ce(outputs, labels)
            loss.backward()
            optimizer_es.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':
    main()
