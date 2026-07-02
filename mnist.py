import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from define import MNISTNet

# 超参数设置
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    """
    加载MNIST数据集
    """
    # 数据预处理：转换为张量并归一化到[0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])

    # 加载训练集（使用本地数据）
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )

    # 加载测试集（使用本地数据）
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, epoch):
    """
    训练模型
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到设备上
        data, target = data.to(DEVICE), target.to(DEVICE)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        output = model(data)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # 每100个批次打印一次
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Average Loss: {avg_loss:.6f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return avg_loss, accuracy

def test(model, test_loader, criterion):
    """
    测试模型
    """
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移动到设备上
            data, target = data.to(DEVICE), target.to(DEVICE)

            # 前向传播
            output = model(data)

            # 计算损失
            test_loss += criterion(output, target).item()

            # 计算预测结果
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy

def plot_results(train_losses, train_accs, test_losses, test_accs):
    """
    绘制训练结果图表
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 保存图表
    plt.savefig('mnist_results.png')
    plt.show()

def main():
    print(f'Using device: {DEVICE}')
    
    # 加载数据
    print('Loading data...')
    train_loader, test_loader = load_data()
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    # 创建模型
    model = MNISTNet().to(DEVICE)
    print(f'\nModel architecture:\n{model}')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录训练过程
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # 训练模型
    print('\nStarting training...')
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/mnist_net.pth')
    print('Model saved to models/mnist_net.pth')

    # 绘制结果
    plot_results(train_losses, train_accs, test_losses, test_accs)

if __name__ == '__main__':
    main()