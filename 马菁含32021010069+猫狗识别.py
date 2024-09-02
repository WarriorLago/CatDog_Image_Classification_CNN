import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import time
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU可用，使用GPU")
else:
    device = torch.device("cpu")
    print("GPU不可用，使用CPU")

# 数据加载与预处理
def load_dataset(directory, batch_size=32, is_train=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 将图片统一调整为64x64
        transforms.ToTensor(),  # 将图片转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
    ])

    dataset = ImageFolder(directory, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader

# 训练集和测试集的加载
train_loader = load_dataset("catdog/training_set")


test_loader = load_dataset("catdog/test_set", is_train=False)


# 定义CNN模型
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # 新增卷积层
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # 调整全连接层参数
        self.fc2 = nn.Linear(512, 256)  # 新增全连接层
        self.fc3 = nn.Linear(256, 2)  # 输出层
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))  # 通过新增的卷积层
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(self.dropout(x)))
        x = torch.relu(self.fc2(self.dropout(x)))  # 通过新增的全连接层
        x = self.fc3(x)
        return x


# 实例化模型并移至GPU（如果可用）
model = CatDogCNN().to(device)
optimizer = optim.Adam(model.parameters())

# 训练模型
def train_model(model, train_loader, optimizer, epochs=20):
    model.train()
    total_batches = len(train_loader)
    loss_values = []  # 初始化一个列表来存储损失值

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            # 存储损失值
            loss_values.append(loss.item())

            # 打印损失信息
            if i % 30 == 0:
                print(f'训练的代数: {epoch+1}/{epochs}, 当前完成: {i}/{total_batches}, 损失函数值: {loss.item()}')

    return loss_values

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'在测试集上的预测准确率: {100 * correct / total}%')


# 训练和评估
start_time = time.time()
loss_values = train_model(model, train_loader, optimizer, epochs=20)
end_time = time.time()
print(f"训练完成，耗时: {end_time - start_time}秒")

# 绘制损失变化图
plt.plot(loss_values)
plt.title('Training Loss')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()


# 评估模型在测试集上的表现
evaluate_model(model, test_loader)

# 保存模型（可选）
torch.save(model.state_dict(), "catdog_model.pth")
