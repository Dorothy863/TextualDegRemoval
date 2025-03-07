# 帮我写一个测试展示batch normalization的例子
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化层
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # 假设输入图像为32x32

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # 应用批归一化
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        return x

# 示例使用
if __name__ == "__main__":
    model = SimpleCNN()
    print(model)

    # 创建一个随机输入张量（批量大小4，3通道，32x32）
    input_tensor = torch.randn(4, 3, 32, 32)
    output = model(input_tensor)
    print("输出形状:", output.shape)  # 应为 (4, 10)
