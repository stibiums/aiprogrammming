import os
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 检测并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 8
n_epochs = 10

data_dir = './data/cifar-10-batches-py'
download = not os.path.exists(data_dir)

cifar10_train = datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)

# 创建DataLoader的函数，确保每次都有相同的数据顺序
def create_dataloaders():
    torch.manual_seed(42)  # 确保数据加载顺序一致
    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

cifar10_train_loader, cifar10_test_loader = create_dataloaders()

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

def train_with_momentum(momentum_value, run_name):
    """使用指定的momentum值训练模型"""
    print(f"\n开始训练 momentum={momentum_value}")

    # 为每个实验设置相同的随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 创建相同的DataLoader
    train_loader, test_loader = create_dataloaders()

    # 创建新的模型实例
    net = cifarNet()
    net.to(device)

    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=momentum_value)

    # TensorBoard writer
    writer = SummaryWriter(f'runs/{run_name}')

    # 存储损失值用于绘图
    epoch_losses = []

    # 训练过程
    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        net.train()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累积损失（在权重更新前）
            epoch_loss += loss.item()
            running_loss += loss.item()

            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        print(f'Epoch [{epoch+1}/{n_epochs}] Average Loss: {avg_epoch_loss:.4f}')

    writer.close()

    # 测试模型
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Momentum {momentum_value} - Test Accuracy: {accuracy:.2f}%')

    return epoch_losses, accuracy

# 测试不同的momentum值
momentum_values = [0.0, 0.5, 0.9, 0.99]
all_losses = {}
accuracies = {}

for momentum in momentum_values:
    run_name = f'momentum_{momentum}'
    losses, accuracy = train_with_momentum(momentum, run_name)
    all_losses[momentum] = losses
    accuracies[momentum] = accuracy

# 绘制Loss曲线对比图
plt.figure(figsize=(10, 6))
for momentum in momentum_values:
    plt.plot(range(1, n_epochs + 1), all_losses[momentum],
             label=f'momentum={momentum} (acc: {accuracies[momentum]:.2f}%)', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Loss Curves with Different Momentum Values')
plt.legend()
plt.grid(True)
plt.savefig('momentum_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 实验结果总结 ===")
for momentum in momentum_values:
    print(f"Momentum {momentum}: 最终准确率 {accuracies[momentum]:.2f}%")