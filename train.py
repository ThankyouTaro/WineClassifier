import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = "train"
test_dir = "test"

# 将图像调整为224×224尺寸并归一化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_set = datasets.ImageFolder(train_dir, transform=train_augs)
test_set = datasets.ImageFolder(test_dir, transform=test_augs)

batch_size = 32
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_set, batch_size=batch_size)


def train(net, train_iter, test_iter, criterion, optimizer, num_epochs):
    net = net.to(device)
    print("training on", device)
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # 训练模式
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # 梯度清零
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        with torch.no_grad():
            net.eval()  # 评估模式
            test_acc_sum, n2 = 0.0, 0
            for X, y in test_iter:
                test_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                n2 += y.shape[0]

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc_sum / n2, time.time() - start))


pretrained_net = models.resnet18(pretrained=True)
num_ftrs = pretrained_net.fc.in_features
pretrained_net.fc = nn.Linear(num_ftrs, 3)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                      lr=lr, weight_decay=0.001)

loss = torch.nn.CrossEntropyLoss()
train(pretrained_net, train_iter, test_iter, loss, optimizer, num_epochs=5)
