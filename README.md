# LeNet识别手写数字0~9
## 需要环境支持
python==3.9  
numpy  
tensorboardX  
pytorch gpu版  
tqdm  
pillow  
## 文件使用
train.py为训练模型文件，用于模型训练、保存、损失值打印、log日志记录  
model.py为模型文件，搭建LeNet模型  
test.py为测试文件，使用models文件夹中训练好的模型（自己挑选）对images中的十张图片进行测试并打印结果  
configer.py为配置文件，所有超参数在其中配置  
dataset.py为数据集测试文件，没用
## 验证集结果
精度达到98%  
