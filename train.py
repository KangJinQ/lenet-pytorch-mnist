import os
import numpy as np
import torchvision.transforms
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
import torch
# 记录信息
from tqdm import tqdm
from tensorboardX import SummaryWriter
# 配置文件
from configer import Config
# 模型导入
from model import MyLeNet


if __name__ == '__main__':
    config = Config()
    work_dir = os.getcwd()
    writer = SummaryWriter('{}/logs'.format(work_dir))

    train_dataset = MNIST(root='./train', train=True, transform=torchvision.transforms.ToTensor())
    test_dataset = MNIST(root='./test', train=False, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
    # 寻找运算设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 定义模型实体
    model = MyLeNet()
    model = model.to(device=device)
    # 随机梯度下降
    sgd = SGD(model.parameters(), lr=config.learning_rate)
    # 损失函数
    loss_fn = CrossEntropyLoss()
    loss_fn = loss_fn.to(device=device)

    for epoch in range(config.all_epochs):
        # 一轮训练
        model.train()
        with tqdm(total=len(train_dataloader), desc=f'epoch:{epoch}') as pbar:
            for idx, (X, y) in enumerate(train_dataloader):
                X = X.to(device)
                y = y.to(device)
                sgd.zero_grad()
                predict_y = model(X.float())
                loss = loss_fn(predict_y, y.long())

                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})
                pbar.update(1)
                if idx % 50 == 0:
                    total_idx = len(train_dataloader) * epoch + idx + 1
                    writer.add_scalar('train_loss', loss, total_idx)
                loss.backward()
                sgd.step()

        # 一轮验证
        model.eval()
        total_test_loss = 0
        all_correct_num = 0
        all_sample_num = 0
        with torch.no_grad():
            with tqdm(total=len(test_dataloader), desc=f'test epoch: {epoch}') as test_pbar:
                for idx, (testX, testy) in enumerate(test_dataloader):
                    testX = testX.to(device)
                    testy = testy.to(device)
                    predict_y = model(testX.float())  # [batch_size, 10]
                    # 记录log
                    loss = loss_fn(predict_y, testy)
                    total_test_loss += loss
                    test_pbar.set_postfix({'loss': '{0:1.5f}'.format(total_test_loss)})
                    test_pbar.update(1)
                    # 统计正确样本数量和总样本数量
                    predict_y = predict_y.argmax(-1)  # [batch_size, 1] or [batch_size,]? ==>(256,)
                    current_correct = predict_y == testy
                    current_correct = current_correct.to('cpu')
                    all_correct_num += np.sum(current_correct.numpy(), -1)
                    all_sample_num += current_correct.shape[0]
                    # print('test_loss is: ', total_test_loss)
                acc = all_correct_num / all_sample_num
                print('accuracy: {:.2f}'.format(acc))
                torch.save(model, './models/mnist_{:.2f}.pkl'.format(acc))
            writer.add_scalar("test_loss", total_test_loss, epoch)
