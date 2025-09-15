import os
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4
n_epochs = 10

data_dir = './data/cifar-10-batches-py'
download = not os.path.exists(data_dir)

cifar10_train = datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)

cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2)
cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class cifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建Tensorboard writer
writer = SummaryWriter('runs/cifar_experiment')

Net = cifarNet()
Loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)

# 训练过程
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(cifar10_train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = Net(inputs)
        loss = Loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # 记录损失到Tensorboard
        global_step = epoch * len(cifar10_train_loader) + i
        writer.add_scalar('Training Loss', loss.item(), global_step)
        
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

save_path = './cifar_net.pth'
torch.save(Net.state_dict(), save_path)

############################################################################

# 测试模型
Net.eval()

# 统计各类别的正确预测数和总数
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

correct = 0
total = 0

with torch.no_grad():
    for data in cifar10_test_loader:
        images, labels = data
        outputs = Net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 统计各类别准确率
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 打印总体准确率
print('Overall Test Accuracy: %.2f%%' % (100 * correct / total))

# 打印各类别准确率
print('\nAccuracy for each class:')
for i in range(10):
    if class_total[i] > 0:
        print('%s: %.2f%%' % (CLASSES[i], 100 * class_correct[i] / class_total[i]))
    else:
        print('%s: N/A (no examples)' % CLASSES[i])

writer.close()
print('\nTensorboard logs saved to runs/cifar_experiment')
print('Run "tensorboard --logdir=runs" to view the loss curve')




