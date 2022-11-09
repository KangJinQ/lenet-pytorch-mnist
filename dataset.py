import torchvision
import numpy as np

if __name__ == '__main__':
    train_dataset = torchvision.datasets.MNIST(root='./train', train=True, download=False)
    test_dataset = torchvision.datasets.MNIST(root='./test', train=False, download=False)
    print(len(train_dataset), len(test_dataset))
    sample = test_dataset[0]
    print(sample)
    img = sample[0]
    img_array = np.array(img).astype(np.float32)
    print(img_array)

