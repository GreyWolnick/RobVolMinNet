import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tools
# import data_load  # File used to create these data sets


learning_rate = 0.01
momentum = 0.5
batch_size_train = 64
batch_size_test = 1000
batch_size = 128  # From anchor points paper
n_epochs = 60

save_dir, model_dir, matrix_dir, logs = create_dir()  # No args since I am just using set numbers for MNIST
# This is from utils.py, needs to be retrofitted with args to work


class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)

    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x)


device = torch.device("cpu")
network = Net()
# criterion = nn.CrossEntropyLoss()  # This is from old MNIST implementation
trans = sig_t(device, 10)  # New initializer T
optimizer_trans = optim.Adam(trans.parameters(), lr=learning_rate, weight_decay=0)
# What is the difference between optim and loss function
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)  # Old optim

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)  # num_workers=2

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

val_loss_list = []
val_acc_list = []
test_acc_list = []

#optimizer and StepLR
optimizer_es = optim.SGD(network.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
scheduler1 = MultiStepLR(optimizer_es, milestones=[], gamma=0.1)  # Empty list to designate no milestones
scheduler2 = MultiStepLR(optimizer_trans, milestones=[], gamma=0.1)

loss_func_ce = F.nll_loss  # Same as criterion on line 62?

print(trainset.t, file=logs, flush=True)  # What is this .t?


t = trans()
est_T = t.detach().cpu().numpy()
print(est_T, file=logs, flush=True)


estimate_error = tools.error(est_T, trainset.t)  # Same question

print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)


if __name__ == '__main__':
    for epoch in range(n_epochs):  # n_epoch = 60

        print('epoch {}'.format(epoch + 1), file=logs, flush=True)
        network.train()  # What is the train method?
        trans.train()

        train_loss = 0.
        train_vol_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.

        for batch_x, batch_y in trainloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer_es.zero_grad()
            optimizer_trans.zero_grad()

            clean = network(batch_x)

            t = trans()

            out = torch.mm(clean, t)

            vol_loss = t.slogdet().logabsdet

            ce_loss = loss_func_ce(out.log(), batch_y.long())
            loss = ce_loss + 0.0001 * vol_loss  # 0.0001 was args.lam

            train_loss += loss.item()
            train_vol_loss += vol_loss.item()

            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()

            loss.backward()
            optimizer_es.step()
            optimizer_trans.step()

        print(
            'Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / (len(trainset)) * batch_size,
                                                                       train_vol_loss / (
                                                                           len(trainset)) * batch_size,
                                                                       train_acc / (len(trainset))), file=logs,
            flush=True)

        scheduler1.step()
        scheduler2.step()

        with torch.no_grad():
            network.eval()
            trans.eval()
            for batch_x, batch_y in val_loader:  # Why does this have validation data but not my mnist implementation?
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = network(batch_x)
                t = trans()

                out = torch.mm(clean, t)
                loss = loss_func_ce(out.log(), batch_y.long())
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data)) * batch_size,
                                                     val_acc / (len(val_data))), file=logs, flush=True)

        with torch.no_grad():
            network.eval()
            trans.eval()

            for batch_x, batch_y in testloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = network(batch_x)

                loss = loss_func_ce(clean.log(), batch_y.long())
                eval_loss += loss.item()
                pred = torch.max(clean, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(testset)) * batch_size,
                                                          eval_acc / (len(testset))), file=logs, flush=True)

            est_T = t.detach().cpu().numpy()
            estimate_error = tools.error(est_T, trainset.t)

            matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (epoch + 1)
            np.save(matrix_path, est_T)

            print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)
            print(est_T, file=logs, flush=True)

        val_loss_list.append(val_loss / (len(val_data)))
        val_acc_list.append(val_acc / (len(val_data)))
        test_acc_list.append(eval_acc / (len(test_data)))

    val_loss_array = np.array(val_loss_list)
    val_acc_array = np.array(val_acc_list)
    model_index = np.argmin(val_loss_array)
    model_index_acc = np.argmax(val_acc_array)

    matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index + 1)
    final_est_T = np.load(matrix_path)
    final_estimate_error = tools.error(final_est_T, trainset.t)

    matrix_path_acc = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index_acc + 1)
    final_est_T_acc = np.load(matrix_path_acc)
    final_estimate_error_acc = tools.error(final_est_T_acc, trainset.t)

    print("Final test accuracy: %f" % test_acc_list[model_index], file=logs, flush=True)
    print("Final test accuracy acc: %f" % test_acc_list[model_index_acc], file=logs, flush=True)
    print("Final estimation error loss: %f" % final_estimate_error, file=logs, flush=True)
    print("Final estimation error loss acc: %f" % final_estimate_error_acc, file=logs, flush=True)
    print("Best epoch: %d" % model_index, file=logs, flush=True)
    print(final_est_T, file=logs, flush=True)
    logs.close()


    # for epoch in range(2):  # loop over the dataset multiple times  OLD MNIST CODE
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()  # Which optimizer to use in training
    #
    #         # forward + backward + optimize
    #         outputs = network(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #             running_loss = 0.0
    #
    # # PATH = './mnist_net.pth'
    # # torch.save(network.state_dict(), PATH)
    # print('Finished Training')
    #
    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         # calculate outputs by running images through the network
    #         outputs = network(images)
    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
